import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_cm(cm, title, out_path, class_names=("nerf", "real")):
    cm = np.array(cm, dtype=int)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    json_path = Path(r"E:\learn_pytorch\nerf\runs\resnet18_baseline_seed42\result_eval_only.json")
    out_dir = Path(r"E:\learn_pytorch\nerf\results\chapter4\figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    class_to_idx = data.get("class_to_idx", {"nerf": 0, "real": 1})
    # 按 idx 排序得到类名
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    test_cm = data["test_metrics"]["cm"]
    plot_cm(
        test_cm,
        title="Confusion Matrix (FFHQ test)",
        out_path=out_dir / "confusion_matrix_ffhq.png",
        class_names=class_names,
    )

    celeba_cm = data["extra_test_celeba_metrics"]["cm"]
    plot_cm(
        celeba_cm,
        title="Confusion Matrix (CelebA real vs NeRF)",
        out_path=out_dir / "confusion_matrix_celeba.png",
        class_names=class_names,
    )

    print("Saved:")
    print(out_dir / "confusion_matrix_ffhq.png")
    print(out_dir / "confusion_matrix_celeba.png")


if __name__ == "__main__":
    main()