"""
price_pipeline.py
=================
Data pipeline: Crawl -> Clean -> Store (CSV + SQLite) -> Linear Regression (tự cài bằng NumPy)

Cách dùng:
    python price_pipeline.py demo      # chạy thử với dữ liệu giả lập (không cần mạng)
    python price_pipeline.py crawl     # crawl web thật (cần sửa SITE_CONFIG cho đúng trang)
    python price_pipeline.py clean     # làm sạch raw_products.csv -> clean_products.csv + SQLite
    python price_pipeline.py train     # huấn luyện mô hình từ clean_products.csv
    python price_pipeline.py predict --ram 16 --storage 512 --screen 15.6

Cài thư viện:
    pip install numpy pandas requests beautifulsoup4
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path("raw_products.csv")
CLEAN_CSV = Path("clean_products.csv")
DB_PATH = Path("products.db")
MODEL_PATH = Path("model_weights.npz")

FEATURES = ["ram_gb", "storage_gb", "screen_inch"]  # đổi thành thông số bạn cần
TARGET = "price_vnd"

# ---------------------------------------------------------------------------
# 1. CRAWL
# ---------------------------------------------------------------------------
# Mỗi trang web có HTML khác nhau: hãy mở trang > chuột phải > Inspect
# rồi sửa các CSS selector bên dưới cho khớp.
SITE_CONFIG = {
    "base_url": "https://example-shop.vn/laptop?page={page}",  # <-- THAY BẰNG TRANG THẬT
    "max_pages": 5,
    "item": "div.product-item",     # khung bao quanh 1 sản phẩm
    "name": "h3.product-name",      # tên sản phẩm
    "price": "span.product-price",  # giá
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StudentProject/1.0)"}


def parse_products(html: str, cfg: dict) -> list[dict]:
    """Tách danh sách sản phẩm từ 1 trang HTML (tách riêng để dễ test)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for item in soup.select(cfg["item"]):
        name = item.select_one(cfg["name"])
        price = item.select_one(cfg["price"])
        if name and price:
            rows.append({"name": name.get_text(strip=True),
                         "price_raw": price.get_text(strip=True)})
    return rows


def crawl(cfg: dict = SITE_CONFIG, delay: float = 1.5) -> pd.DataFrame:
    """Crawl nhiều trang bằng requests + BeautifulSoup. Nghỉ giữa các request cho lịch sự."""
    import requests

    all_rows = []
    for page in range(1, cfg["max_pages"] + 1):
        url = cfg["base_url"].format(page=page)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[!] Lỗi trang {page}: {e}")
            break
        rows = parse_products(resp.text, cfg)
        print(f"[+] Trang {page}: {len(rows)} sản phẩm")
        if not rows:  # hết sản phẩm
            break
        all_rows.extend(rows)
        time.sleep(delay)

    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"[✓] Đã lưu {len(df)} dòng thô vào {RAW_CSV}")
    return df


# Nếu trang web render bằng JavaScript thì requests không lấy được dữ liệu,
# lúc đó dùng Selenium: lấy driver.page_source rồi đưa vào parse_products().
#   from selenium import webdriver
#   driver = webdriver.Chrome(); driver.get(url); html = driver.page_source


# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------
def parse_price(text: str) -> float:
    """'12.990.000₫' / '12,990,000 VND' -> 12990000.0"""
    digits = re.sub(r"[^\d]", "", str(text))
    return float(digits) if digits else np.nan


def extract_specs(name: str) -> dict:
    """Trích thông số kỹ thuật từ tên sản phẩm bằng regex."""
    name = str(name)
    ram = re.search(r"(\d+)\s*GB\s*RAM|RAM\s*(\d+)\s*GB|(\d+)\s*GB\s*/", name, re.I)
    ssd = re.search(r"(\d+)\s*(TB|GB)\s*(?:SSD|HDD|NVMe)|(?:SSD|HDD)\s*(\d+)\s*(TB|GB)", name, re.I)
    scr = re.search(r"(\d{2}(?:\.\d)?)\s*(?:inch|\"|in\b)", name, re.I)

    ram_gb = next((int(g) for g in ram.groups() if g), np.nan) if ram else np.nan

    storage_gb = np.nan
    if ssd:
        g = [x for x in ssd.groups() if x]
        storage_gb = int(g[0]) * (1024 if g[1].upper() == "TB" else 1)

    return {
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "screen_inch": float(scr.group(1)) if scr else np.nan,
    }


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df[TARGET] = df["price_raw"].map(parse_price)
    specs = df["name"].map(extract_specs).apply(pd.Series)
    df = pd.concat([df, specs], axis=1)

    n0 = len(df)
    df = df.drop_duplicates(subset="name")
    df = df.dropna(subset=[TARGET] + FEATURES)

    # Loại outlier giá bằng quy tắc IQR
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[df[TARGET].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

    print(f"[✓] Làm sạch: {n0} -> {len(df)} dòng")
    return df.reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("products", conn, if_exists="replace", index=False)
    print(f"[✓] Đã lưu {CLEAN_CSV} và bảng 'products' trong {DB_PATH}")


# ---------------------------------------------------------------------------
# 3. MÔ HÌNH: LINEAR REGRESSION TỰ CÀI (ĐẠI SỐ TUYẾN TÍNH + GIẢI TÍCH)
# ---------------------------------------------------------------------------
class LinearRegression:
    """
    Mô hình:   ŷ = X w        (X đã thêm cột 1 cho hệ số chặn b)

    Hàm mất mát (MSE):   J(w) = (1/m) ||X w - y||²

    ── Giải tích 1 (đạo hàm) ──
    Đạo hàm theo w:      ∇J(w) = (2/m) Xᵀ (X w - y)
    Gradient Descent:    w ← w - α ∇J(w)

    ── Đại số tuyến tính ──
    Cho ∇J = 0  =>  XᵀX w = Xᵀ y   (phương trình chuẩn - normal equation)
                =>  w = (XᵀX)⁻¹ Xᵀ y
    """

    def __init__(self):
        self.w = None      # [b, w1, w2, ...] trên dữ liệu đã chuẩn hoá
        self.mu = None
        self.sigma = None
        self.y_mu = 0.0
        self.y_sigma = 1.0
        self.loss_history = []

    # --- tiện ích ---
    def _scale(self, X):
        return (X - self.mu) / self.sigma

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _fit_scaler(self, X, y):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma[self.sigma == 0] = 1.0
        self.y_mu, self.y_sigma = y.mean(), y.std() or 1.0

    # --- cách 1: phương trình chuẩn (đại số tuyến tính) ---
    def fit_normal_equation(self, X, y, ridge=1e-8):
        self._fit_scaler(X, y)
        A = self._add_bias(self._scale(X))
        t = (y - self.y_mu) / self.y_sigma
        # giải (AᵀA + λI) w = Aᵀ t  -- dùng solve ổn định hơn tính nghịch đảo trực tiếp
        n = A.shape[1]
        self.w = np.linalg.solve(A.T @ A + ridge * np.eye(n), A.T @ t)
        return self

    # --- cách 2: gradient descent (giải tích) ---
    def fit_gradient_descent(self, X, y, lr=0.1, epochs=1000):
        self._fit_scaler(X, y)
        A = self._add_bias(self._scale(X))
        t = (y - self.y_mu) / self.y_sigma
        m = A.shape[0]
        self.w = np.zeros(A.shape[1])
        self.loss_history = []
        for _ in range(epochs):
            err = A @ self.w - t
            self.loss_history.append((err @ err) / m)   # J(w)
            grad = (2 / m) * A.T @ err                  # ∇J(w)
            self.w -= lr * grad                         # bước cập nhật
        return self

    def predict(self, X):
        X = np.atleast_2d(X).astype(float)
        A = self._add_bias(self._scale(X))
        return (A @ self.w) * self.y_sigma + self.y_mu

    def coefficients(self, names):
        """Hệ số quy đổi về đơn vị gốc: giá thay đổi bao nhiêu khi tăng 1 đơn vị đặc trưng."""
        slopes = self.w[1:] * self.y_sigma / self.sigma
        intercept = self.y_mu + self.w[0] * self.y_sigma - np.sum(slopes * self.mu)
        return intercept, dict(zip(names, slopes))

    def save(self, path=MODEL_PATH):
        np.savez(path, w=self.w, mu=self.mu, sigma=self.sigma,
                 y_mu=self.y_mu, y_sigma=self.y_sigma)

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = np.load(path)
        m = cls()
        m.w, m.mu, m.sigma = d["w"], d["mu"], d["sigma"]
        m.y_mu, m.y_sigma = float(d["y_mu"]), float(d["y_sigma"])
        return m


def r2_score(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y, y_hat):
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def train(df: pd.DataFrame, seed: int = 42) -> LinearRegression:
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)

    # chia train/test 80/20
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = idx[:cut], idx[cut:]

    ne = LinearRegression().fit_normal_equation(X[tr], y[tr])
    gd = LinearRegression().fit_gradient_descent(X[tr], y[tr])

    print("\n=== KẾT QUẢ ===")
    for name, mdl in [("Normal Equation", ne), ("Gradient Descent", gd)]:
        p = mdl.predict(X[te])
        print(f"{name:<17} R²={r2_score(y[te], p):.4f}  RMSE={rmse(y[te], p):,.0f} VND")

    print(f"\nChênh lệch trọng số 2 phương pháp: {np.linalg.norm(ne.w - gd.w):.2e}"
          "  (gần 0 => gradient descent đã hội tụ về nghiệm của phương trình chuẩn)")
    print(f"Loss đầu: {gd.loss_history[0]:.4f} -> loss cuối: {gd.loss_history[-1]:.4f}")

    b, slopes = ne.coefficients(FEATURES)
    print(f"\nPhương trình: giá = {b:,.0f}", end="")
    for k, v in slopes.items():
        print(f" + {v:,.0f}·{k}", end="")
    print()

    ne.save()
    print(f"[✓] Đã lưu mô hình vào {MODEL_PATH}")
    return ne


# ---------------------------------------------------------------------------
# DEMO: dữ liệu giả lập để chạy thử khi chưa crawl được
# ---------------------------------------------------------------------------
def make_demo_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ram = int(rng.choice([8, 16, 32]))
        ssd = int(rng.choice([256, 512, 1024]))
        scr = float(rng.choice([13.3, 14, 15.6, 16]))
        price = 5e6 + ram * 4.2e5 + ssd * 6e3 + scr * 3e5 + rng.normal(0, 8e5)
        price_text = f"{price:,.0f}₫".replace(",", ".")
        rows.append({"name": f"Laptop Model-{i} {scr} inch {ram}GB RAM {ssd}GB SSD",
                     "price_raw": price_text})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["demo", "crawl", "clean", "train", "predict"])
    ap.add_argument("--ram", type=float)
    ap.add_argument("--storage", type=float)
    ap.add_argument("--screen", type=float)
    a = ap.parse_args()

    if a.cmd == "demo":
        raw = make_demo_raw()
        raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
        df = clean(raw)
        save(df)
        train(df)
    elif a.cmd == "crawl":
        crawl()
    elif a.cmd == "clean":
        save(clean(pd.read_csv(RAW_CSV)))
    elif a.cmd == "train":
        train(pd.read_csv(CLEAN_CSV))
    elif a.cmd == "predict":
        if None in (a.ram, a.storage, a.screen):
            ap.error("predict cần --ram --storage --screen")
        model = LinearRegression.load()
        price = model.predict([a.ram, a.storage, a.screen])[0]
        print(f"Giá dự đoán: {price:,.0f} VND")


if __name__ == "__main__":
    main()
