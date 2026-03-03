import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mannwhitneyu

ROOT = Path(r"E:\learn_pytorch\nerf\results")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ====== 改成你真实数据目录（与之前保持一致）======
NERF_DIRS = {
    "nerf_lego": r"E:\learn_pytorch\nerf\dataset\nerf_fake\lego",
    "nerf_chair": r"E:\learn_pytorch\nerf\dataset\nerf_fake\chair",
    "nerf_drums": r"E:\learn_pytorch\nerf\dataset\nerf_fake\drums",
    # "nerf_tiny_lego": r"E:\learn_pytorch\nerf\dataset\nerf_fake\tiny_lego",
}
REAL_DIR = r"E:\learn_pytorch\nerf\dataset\real\ffhq"
# ===================================================

RADIUS_RATIO = 0.25  # 论文里就写这个

def high_freq_power_ratio(img_gray, radius_ratio=0.25):
    """
    高频功率占比（power ratio）
    - 输入：灰度图 float32
    - 输出：高频功率 / 总功率
    """
    # 去均值 + 标准化（让不同图亮度对比度影响更小）
    img = img_gray.astype(np.float32)
    img = img - img.mean()
    std = img.std()
    if std > 1e-8:
        img = img / std

    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    power = (np.abs(fft_shift) ** 2).astype(np.float64)

    h, w = power.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * radius_ratio)

    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    low = power[mask_low].sum()
    total = power.sum()
    high = total - low

    if total <= 1e-12:
        return 0.0
    return float(high / total)

def process_folder(folder_path, label):
    folder = Path(folder_path)
    rows = []
    for fname in tqdm(os.listdir(folder), desc=label):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            p = folder / fname
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ratio = high_freq_power_ratio(img, radius_ratio=RADIUS_RATIO)
            rows.append({"image": fname, "label": label, "high_freq_power_ratio": ratio})
    return rows

def summarize_and_test(df):
    df = df.copy()
    df["binary"] = df["label"].apply(lambda x: "nerf" if str(x).startswith("nerf") else "real")

    nerf = df[df["binary"] == "nerf"]["high_freq_power_ratio"].to_numpy()
    real = df[df["binary"] == "real"]["high_freq_power_ratio"].to_numpy()

    stat, p = mannwhitneyu(nerf, real, alternative="two-sided")

    # 打印到终端
    print("\n[high_freq_power_ratio]")
    print(f"radius_ratio={RADIUS_RATIO}")
    print(f"nerf: mean={nerf.mean():.6f} std={nerf.std():.6f} n={len(nerf)}")
    print(f"real: mean={real.mean():.6f} std={real.std():.6f} n={len(real)}")
    print(f"Mann-Whitney U p-value={p:.3e}")

    # 保存到 CSV（表4-1可以直接引用）
    stats = pd.DataFrame([{
        "feature": "high_freq_power_ratio",
        "radius_ratio": RADIUS_RATIO,
        "nerf_n": int(len(nerf)),
        "nerf_mean": float(nerf.mean()),
        "nerf_std": float(nerf.std()),
        "real_n": int(len(real)),
        "real_mean": float(real.mean()),
        "real_std": float(real.std()),
        "test": "Mann-Whitney U",
        "p_value": float(p),
    }])

    out_stats = TABLES / "high_freq_power_ratio_stats.csv"
    stats.to_csv(out_stats, index=False)
    return out_stats

def boxplot(df, out_png):
    order = sorted(df["label"].unique())
    data = [df[df["label"] == lab]["high_freq_power_ratio"].to_numpy() for lab in order]

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, tick_labels=order, showfliers=True)
    plt.ylabel("high_freq_power_ratio")
    plt.title(f"High-frequency power ratio (radius_ratio={RADIUS_RATIO})")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    rows = []
    for label, path in NERF_DIRS.items():
        rows.extend(process_folder(path, label))
    rows.extend(process_folder(REAL_DIR, "real_ffhq"))

    df = pd.DataFrame(rows)

    out_csv = TABLES / "high_freq_power_ratio.csv"
    df.to_csv(out_csv, index=False)

    out_stats = summarize_and_test(df)

    out_png = FIGS / "box_high_freq_power_ratio.png"
    boxplot(df, out_png)

    print("\nSaved:")
    print(" -", out_stats)
    print(" -", out_csv)
    print(" -", out_png)

if __name__ == "__main__":
    main()