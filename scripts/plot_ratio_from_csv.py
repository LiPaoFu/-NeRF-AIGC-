import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

ROOT = Path(r"E:\learn_pytorch\nerf\results")
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

def load_metric(csv_path, metric_name):
    df = pd.read_csv(csv_path)
    # csv 里列名是 high_freq_ratio，但 mid_high 那个其实也是这个列名
    df = df.rename(columns={"high_freq_ratio": metric_name})
    return df[["label", metric_name]]

def summarize_and_test(df, metric_name):
    df = df.copy()
    df["binary"] = df["label"].apply(lambda x: "nerf" if str(x).startswith("nerf") else "real")

    nerf = df[df["binary"] == "nerf"][metric_name].to_numpy()
    real = df[df["binary"] == "real"][metric_name].to_numpy()

    stat, p = mannwhitneyu(nerf, real, alternative="two-sided")

    print(f"\n[{metric_name}]")
    print(f"nerf: mean={nerf.mean():.6f} std={nerf.std():.6f} n={len(nerf)}")
    print(f"real: mean={real.mean():.6f} std={real.std():.6f} n={len(real)}")
    print(f"Mann-Whitney U p-value={p:.3e}")

def boxplot_by_label(df, metric_name, out_png):
    order = sorted(df["label"].unique())
    data = [df[df["label"] == lab][metric_name].to_numpy() for lab in order]

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, tick_labels=order, showfliers=True)
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    hf = load_metric(TABLES / "high_freq_energy_ratio.csv", "high_freq_ratio")
    mhf = load_metric(TABLES / "mid_high_freq_energy_ratio.csv", "mid_high_freq_ratio")

    summarize_and_test(hf, "high_freq_ratio")
    summarize_and_test(mhf, "mid_high_freq_ratio")

    boxplot_by_label(hf, "high_freq_ratio", FIGS / "box_high_freq_ratio.png")
    boxplot_by_label(mhf, "mid_high_freq_ratio", FIGS / "box_mid_high_freq_ratio.png")

    print("\nSaved:")
    print(" -", FIGS / "box_high_freq_ratio.png")
    print(" -", FIGS / "box_mid_high_freq_ratio.png")

if __name__ == "__main__":
    main()