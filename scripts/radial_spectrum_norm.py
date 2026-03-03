import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


ROOT = Path(r"E:\learn_pytorch\nerf\results")
FIGS = ROOT / "figures"
TABLES = ROOT / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

NERF_DIRS = {
    "nerf_lego": r"E:\learn_pytorch\nerf\dataset\nerf_fake\lego",
    "nerf_chair": r"E:\learn_pytorch\nerf\dataset\nerf_fake\chair",
    "nerf_drums": r"E:\learn_pytorch\nerf\dataset\nerf_fake\drums",
}
REAL_DIR = r"E:\learn_pytorch\nerf\dataset\real\ffhq"
# ============================================================


def load_gray_float(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    img = img.astype(np.float32)

    # 关键：去均值 + 标准化（避免亮度/对比度影响）
    img = img - img.mean()
    std = img.std()
    if std > 1e-8:
        img = img / std
    return img


def radial_power_spectrum_norm(img, num_bins=50):
    """
    规范版径向功率谱：
    - 使用功率谱 |F|^2
    - 输出每张图的径向谱，并做归一化（sum=1）
    """
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r / (r.max() + 1e-8)  # [0,1]

    bins = np.linspace(0, 1, num_bins + 1)
    radial = np.zeros(num_bins, dtype=np.float64)

    for i in range(num_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            radial[i] = power[mask].mean()

    # 关键：每张图归一化成分布，便于跨图比较
    s = radial.sum()
    if s > 1e-12:
        radial = radial / s
    return radial


def process_folder(folder_path, label, num_bins=50):
    folder = Path(folder_path)
    results = []
    for fname in tqdm(os.listdir(folder), desc=label):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            p = folder / fname
            img = load_gray_float(p)
            spectrum = radial_power_spectrum_norm(img, num_bins=num_bins)
            results.append({"image": fname, "label": label, "spectrum": spectrum})
    return results


def main():
    num_bins = 50

    all_results = []
    for label, path in NERF_DIRS.items():
        all_results.extend(process_folder(path, label, num_bins=num_bins))
    all_results.extend(process_folder(REAL_DIR, "real_ffhq", num_bins=num_bins))

    # 聚合：每类求均值和标准差
    spectra = {}
    for item in all_results:
        spectra.setdefault(item["label"], []).append(item["spectrum"])

    mean_spectra = {k: np.mean(v, axis=0) for k, v in spectra.items()}
    std_spectra = {k: np.std(v, axis=0) for k, v in spectra.items()}

    # 保存 CSV（均值谱）
    df = pd.DataFrame({k: mean_spectra[k] for k in sorted(mean_spectra.keys())})
    out_csv = TABLES / "radial_spectrum_avg_norm.csv"
    df.to_csv(out_csv, index=False)

    # ====== 画图：全范围 + 放大前10个bin ======
    x = np.arange(num_bins)
    labels = sorted(mean_spectra.keys())

    def plot_range(x_max, out_name, title_suffix):
        plt.figure(figsize=(9, 5))
        for label in labels:
            m = mean_spectra[label]
            s = std_spectra[label]
            plt.plot(x[:x_max], m[:x_max], label=label, linewidth=2)
            plt.fill_between(x[:x_max], (m - s)[:x_max], (m + s)[:x_max], alpha=0.15)

        plt.xlabel("Normalized Frequency Radius Bin")
        plt.ylabel("Normalized Radial Power (sum=1)")
        plt.title(f"NeRF vs FFHQ Normalized Radial Power Spectrum {title_suffix}")
        plt.legend()
        plt.tight_layout()

        out_png = FIGS / out_name
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(" -", out_png)

    print("\nSaved figures:")
    plot_range(num_bins, "radial_spectrum_comparison_norm_full.png", "(Full)")
    plot_range(10, "radial_spectrum_comparison_norm_zoom10.png", "(Zoom: first 10 bins)")
    # ====== 额外输出：只看 bin1~10（去掉 bin0，差异更明显） ======
    x = np.arange(num_bins)
    labels = sorted(mean_spectra.keys())

    x_start = 1
    x_end = 11  # 1..10

    plt.figure(figsize=(9, 5))
    for label in labels:
        m = mean_spectra[label]
        s = std_spectra[label]
        plt.plot(x[x_start:x_end], m[x_start:x_end], label=label, linewidth=2)
        plt.fill_between(
            x[x_start:x_end],
            (m - s)[x_start:x_end],
            (m + s)[x_start:x_end],
            alpha=0.15
        )

    plt.xlabel("Normalized Frequency Radius Bin")
    plt.ylabel("Normalized Radial Power (sum=1)")
    plt.title("NeRF vs FFHQ Normalized Radial Power Spectrum (Zoom: bins 1-10)")
    plt.legend()
    plt.tight_layout()

    out_png = FIGS / "radial_spectrum_comparison_norm_zoom_bins1_10.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(" -", out_png)

    print(" -", out_csv)
if __name__ == "__main__":
    main()