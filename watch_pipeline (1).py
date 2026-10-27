"""
watch_pipeline.py  -  Crawl donghotot.com.vn -> Clean -> CSV + SQLite -> Linear Regression

Cách dùng:
    python watch_pipeline.py demo       # chạy thử bằng HTML giả lập (không cần mạng)
    python watch_pipeline.py crawl      # crawl web thật
    python watch_pipeline.py clean
    python watch_pipeline.py train
    python watch_pipeline.py predict --brand Longines --movement automatic --gender nam --diameter 40

Cài thư viện:
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

NO_SHOW = False  # đặt bằng --no-show để chỉ lưu PNG, không bật cửa sổ
RAW_CSV = Path("watch_raw.csv")
CLEAN_CSV = Path("watch_clean.csv")
DB_PATH = Path("watches.db")
MODEL_PATH = Path("watch_model.npz")

# ---------------------------------------------------------------------------
# 1. CRAWL  (không phụ thuộc class CSS: nhận diện thẻ sản phẩm qua link /Dong-ho-Nam-... /Dong-ho-Nu-...)
# ---------------------------------------------------------------------------
BASE = "https://www.donghotot.com.vn"
# Các trang danh sách cần crawl (thêm bớt tuỳ ý). Trang chủ đã có ~250 sản phẩm.
START_URLS = [
    BASE + "/",
    BASE + "/Dong-Ho-Nam",
    BASE + "/Dong-Ho-Nu",
    BASE + "/Longines",
    BASE + "/Tissot",
    BASE + "/Mido",
    BASE + "/Hamilton",
    BASE + "/Citizen",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StudentProject/1.0)"}
PRODUCT_RE = re.compile(r"/Dong-ho-(Nam|Nu)-", re.I)


def _find_card(anchor):
    """Đi ngược lên cây HTML, lấy khối lớn nhất mà chỉ chứa ĐÚNG 1 sản phẩm."""
    card, node = anchor, anchor
    while node.parent is not None:
        node = node.parent
        hrefs = {a["href"] for a in node.find_all("a", href=PRODUCT_RE)}
        if len(hrefs) > 1:
            break
        card = node
    return card


def parse_products(html: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows, seen = [], set()
    for a in soup.find_all("a", href=PRODUCT_RE):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        lines = [l for l in _find_card(a).get_text("\n", strip=True).split("\n") if l]
        text = "\n".join(lines)
        if "Giá:" not in text:
            continue

        name_idx = next((i for i, l in enumerate(lines) if l.startswith("Đồng hồ ")), None)
        if name_idx is None:
            continue
        mv = re.search(r"^(Automatic|Quartz|Mechanical|Eco-Drive)[^\n]*", text, re.M | re.I)
        dia = re.search(r"Kích thước mặt số:\s*([^\n]+)", text)
        price = re.search(r"Giá:\s*([\d.,]+)\s*đ", text)
        rows.append({
            "url": href if href.startswith("http") else BASE + href,
            "name": lines[name_idx],
            "collection": lines[name_idx - 1] if name_idx > 0 else "",
            "movement_raw": mv.group(1) if mv else "",
            "diameter_raw": dia.group(1) if dia else "",
            "price_raw": price.group(1) if price else "",
        })
    return rows


def crawl(urls=START_URLS, delay: float = 2.0) -> pd.DataFrame:
    import requests

    rows = []
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[!] {url}: {e}")
            continue
        found = parse_products(r.text)
        print(f"[+] {url}: {len(found)} sản phẩm")
        rows.extend(found)
        time.sleep(delay)
    df = pd.DataFrame(rows).drop_duplicates(subset="url")
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"[✓] Lưu {len(df)} sản phẩm thô vào {RAW_CSV}")
    return df


# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)

    df["price_vnd"] = pd.to_numeric(df["price_raw"].astype(str).str.replace(r"[^\d]", "", regex=True),
                                    errors="coerce")
    # lấy số đầu tiên: "39.2 x 40mm" -> 39.2 ; "29.60 mm" -> 29.6
    df["diameter_mm"] = pd.to_numeric(df["diameter_raw"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
                                      errors="coerce")
    # "Đồng hồ Nam Longines L3.788..." -> gender, brand
    parts = df["name"].str.extract(r"Đồng hồ\s+(Nam|Nữ)\s+(\S+)")
    df["gender"] = parts[0].map({"Nam": "nam", "Nữ": "nu"})
    df["brand"] = parts[1]

    mv = df["movement_raw"].astype(str).str.lower()
    df["movement"] = np.select(
        [mv.str.startswith("automatic"), mv.str.startswith("mechanical"),
         mv.str.startswith("quartz"), mv.str.startswith("eco")],
        ["automatic", "manual", "quartz", "quartz"], default="")

    df = df[df["movement"] != ""]
    df = df.dropna(subset=["price_vnd", "diameter_mm", "gender", "brand"])
    df = df.drop_duplicates(subset="url")

    # loại outlier giá (IQR trên log giá)
    lp = np.log(df["price_vnd"])
    q1, q3 = lp.quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[lp.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

    print(f"[✓] Làm sạch: {n0} -> {len(df)} dòng")
    return df[["url", "name", "collection", "brand", "gender", "movement",
               "diameter_mm", "price_vnd"]].reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("watches", conn, if_exists="replace", index=False)
    print(f"[✓] Đã lưu {CLEAN_CSV} và bảng 'watches' trong {DB_PATH}")


# ---------------------------------------------------------------------------
# 3. FEATURE + MÔ HÌNH
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame):
    """Biến cột chữ (hãng, loại máy, giới tính) thành cột 0/1 (one-hot) để nhân ma trận được."""
    X = pd.get_dummies(df[["brand", "movement", "gender"]], dtype=float)
    # bỏ 1 cột mỗi nhóm làm mốc so sánh (tránh đa cộng tuyến làm XᵀX suy biến)
    drop = [f"brand_{sorted(df['brand'].unique())[0]}", "movement_quartz", "gender_nam"]
    X = X.drop(columns=[c for c in drop if c in X.columns])
    X.insert(0, "diameter_mm", df["diameter_mm"].astype(float))
    return X


class LinearRegression:
    """
    ŷ = X w                         (X có thêm cột 1 cho hệ số chặn)
    J(w) = (1/m) ||X w - y||²       (MSE)
    ∇J(w) = (2/m) Xᵀ (X w - y)      <- đạo hàm riêng theo w   (Giải tích)
    ∇J = 0  =>  (XᵀX) w = Xᵀ y      <- phương trình chuẩn      (Đại số tuyến tính)
    """

    def fit_normal_equation(self, X, y, ridge=1e-6):
        self._scaler(X, y)
        A = self._bias(self._scale(X))
        t = (y - self.y_mu) / self.y_sigma
        self.w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ t)
        return self

    def fit_gradient_descent(self, X, y, lr=0.1, epochs=3000):
        self._scaler(X, y)
        A = self._bias(self._scale(X))
        t = (y - self.y_mu) / self.y_sigma
        m = len(t)
        self.w = np.zeros(A.shape[1])
        self.loss = []
        for _ in range(epochs):
            err = A @ self.w - t
            self.loss.append(float(err @ err) / m)
            self.w -= lr * (2 / m) * A.T @ err
        return self

    def predict(self, X):
        A = self._bias(self._scale(np.atleast_2d(X).astype(float)))
        return (A @ self.w) * self.y_sigma + self.y_mu

    def raw_coefficients(self):
        slopes = self.w[1:] * self.y_sigma / self.sigma
        b = self.y_mu + self.w[0] * self.y_sigma - np.sum(slopes * self.mu)
        return b, slopes

    # tiện ích
    def _scaler(self, X, y):
        self.mu, self.sigma = X.mean(0), X.std(0)
        self.sigma[self.sigma == 0] = 1.0
        self.y_mu, self.y_sigma = y.mean(), (y.std() or 1.0)

    def _scale(self, X):
        return (X - self.mu) / self.sigma

    @staticmethod
    def _bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def plot_loss(loss, loss_opt, path="loss_curve.png", show=True):
    """Vẽ đường cong loss của Gradient Descent so với mức tối ưu của phương trình chuẩn."""
    import matplotlib.pyplot as plt

    zoom = min(100, len(loss))  # GD hội tụ rất nhanh nên phóng to 100 epoch đầu cho dễ nhìn
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, n, title in zip(axes, (zoom, len(loss)), (f"{zoom} epoch đầu (phóng to)", "Toàn bộ, thang log")):
        ax.plot(range(1, n + 1), loss[:n], label="Gradient Descent")
        ax.axhline(loss_opt, color="red", linestyle="--", label="Mức tối ưu (Normal Equation)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss J(w)")
        ax.set_title(title)
        if n == len(loss):
            ax.set_yscale("log")
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Loss của Gradient Descent theo số vòng lặp")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[✓] Đã lưu biểu đồ loss vào {path}")
    if show:
        plt.show()


def train(df: pd.DataFrame, seed: int = 42):
    Xdf = build_features(df)
    names = list(Xdf.columns)
    X = Xdf.to_numpy(float)
    y = np.log(df["price_vnd"].to_numpy(float))  # hồi quy trên log(giá): hãng/loại máy nhân giá lên theo %

    idx = np.random.default_rng(seed).permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = idx[:cut], idx[cut:]

    ne = LinearRegression().fit_normal_equation(X[tr], y[tr])
    gd = LinearRegression().fit_gradient_descent(X[tr], y[tr])

    print(f"\nSố mẫu: train={len(tr)}, test={len(te)}")
    print("=== KẾT QUẢ (tập test) ===")
    for label, mdl in [("Normal Equation", ne), ("Gradient Descent", gd)]:
        p = mdl.predict(X[te])
        real, pred = np.exp(y[te]), np.exp(p)
        mape = np.mean(np.abs(real - pred) / real) * 100
        print(f"{label:<17} R²(log giá)={r2(y[te], p):.4f}   sai số % trung bình={mape:.1f}%")
    print(f"\n|w_normal - w_gd| = {np.linalg.norm(ne.w - gd.w):.2e}  (≈0: hai cách ra cùng nghiệm)")
    print(f"Loss GD: {gd.loss[0]:.4f} -> {gd.loss[-1]:.4f}")
    loss_opt = float(np.mean((ne.predict(X[tr]) - y[tr]) ** 2) / ne.y_sigma ** 2)
    plot_loss(gd.loss, loss_opt, show=not NO_SHOW)

    b, slopes = ne.raw_coefficients()
    print("\nẢnh hưởng lên giá (so với mốc: Quartz, nam, hãng đầu bảng chữ cái):")
    for n, s in zip(names, slopes):
        print(f"  {n:<22} {(np.exp(s) - 1) * 100:+7.1f}%  {'(mỗi +1 mm)' if n == 'diameter_mm' else ''}")

    np.savez(MODEL_PATH, w=ne.w, mu=ne.mu, sigma=ne.sigma, y_mu=ne.y_mu,
             y_sigma=ne.y_sigma, names=np.array(names))
    print(f"\n[✓] Đã lưu mô hình vào {MODEL_PATH}")


def predict(brand, movement, gender, diameter):
    d = np.load(MODEL_PATH)
    names = [str(n) for n in d["names"]]
    m = LinearRegression()
    m.w, m.mu, m.sigma, m.y_mu, m.y_sigma = d["w"], d["mu"], d["sigma"], float(d["y_mu"]), float(d["y_sigma"])
    x = np.zeros(len(names))
    for i, n in enumerate(names):
        if n == "diameter_mm":
            x[i] = diameter
        elif n in (f"brand_{brand}", f"movement_{movement}", f"gender_{gender}"):
            x[i] = 1.0
    price = float(np.exp(m.predict(x)[0]))
    print(f"Giá dự đoán: {price:,.0f} VND")


# ---------------------------------------------------------------------------
# DEMO: sinh HTML giả giống cấu trúc trang thật rồi chạy qua parser + pipeline
# ---------------------------------------------------------------------------
def make_demo_html(n=250, seed=1) -> str:
    rng = np.random.default_rng(seed)
    base = {"Longines": 4.3, "Tissot": 3.2, "Mido": 3.4, "Hamilton": 3.5}
    mv_eff = {"Automatic (Máy cơ tự động lên cót)": 0.45,
              "Mechanical (Máy cơ lên cót thủ công)": 0.5,
              "Quartz (Máy Pin)": 0.0}
    cards = []
    for i in range(n):
        brand = rng.choice(list(base))
        mv = rng.choice(list(mv_eff))
        g = rng.choice(["Nam", "Nữ"])
        dia = float(rng.choice([29, 30, 34, 38, 39, 40, 41, 42]))
        lp = 16.0 + base[brand] * 0.2 + mv_eff[mv] + 0.025 * dia + (-0.05 if g == "Nữ" else 0) + rng.normal(0, 0.12)
        price = int(round(np.exp(lp) / 1000) * 1000)
        price_s = f"{price:,}".replace(",", ".") + "đ"
        href = f"/Dong-ho-{'Nam' if g == 'Nam' else 'Nu'}-{brand}-REF{i}"
        cards.append(f"""<div class="p"><a href="{href}"><img src="x.jpg"></a>
          <div>{brand} Collection {i}</div>
          <a href="{href}">Đồng hồ {g} {brand} REF{i}</a>
          <div>{mv}</div><div>Kích thước mặt số: {dia:.2f} mm</div>
          <div>Giá: {price_s}</div><div>Giá ưu đãi: {price_s}</div></div>""")
    return "<html><body>" + "".join(cards) + "</body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["demo", "crawl", "clean", "train", "predict"])
    ap.add_argument("--brand", default="Longines")
    ap.add_argument("--movement", default="automatic", choices=["automatic", "manual", "quartz"])
    ap.add_argument("--gender", default="nam", choices=["nam", "nu"])
    ap.add_argument("--diameter", type=float, default=40.0)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    global NO_SHOW
    NO_SHOW = a.no_show

    if a.cmd == "demo":
        raw = pd.DataFrame(parse_products(make_demo_html()))
        raw.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
        print(f"[demo] Parser đọc được {len(raw)} sản phẩm từ HTML giả lập")
        df = clean(raw)
        save(df)
        train(df)
    elif a.cmd == "crawl":
        crawl()
    elif a.cmd == "clean":
        save(clean(pd.read_csv(RAW_CSV)))
    elif a.cmd == "train":
        train(pd.read_csv(CLEAN_CSV))
    else:
        predict(a.brand, a.movement, a.gender, a.diameter)


if __name__ == "__main__":
    main()
