import json
from pathlib import Path
import numpy as np

ROOT = Path(r"E:\learn_pytorch\nerf\runs")

# 会匹配：resnet18_baseline_seed1, seed2, ...
run_dirs = sorted([p for p in ROOT.glob("resnet18_baseline_seed*") if p.is_dir()])

if not run_dirs:
    raise SystemExit("No run dirs found. Did you run train_resnet18.py with --seed ?")

rows = []
for d in run_dirs:
    fp = d / "result.json"
    if not fp.exists():
        print(f"[SKIP] missing {fp}")
        continue
    data = json.loads(fp.read_text(encoding="utf-8"))
    seed = data["cfg"].get("seed", None)
    tm = data["test_metrics"]
    rows.append({
        "dir": str(d.name),
        "seed": seed,
        "acc": tm["acc"],
        "precision": tm["precision"],
        "recall": tm["recall"],
        "f1": tm["f1"],
        "auc": tm["auc"],
        "cm": tm["cm"],
    })

if not rows:
    raise SystemExit("No valid result.json found.")

# 打印每次结果
print("\n=== PER-RUN TEST METRICS ===")
for r in rows:
    print(f"{r['dir']} (seed={r['seed']}): "
          f"acc={r['acc']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} cm={r['cm']}")

# 汇总
def mean_std(key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

print("\n=== SUMMARY (mean ± std on TEST) ===")
for k in ["acc", "precision", "recall", "f1", "auc"]:
    m, s = mean_std(k)
    print(f"{k}: {m:.4f} ± {s:.4f}")

# 保存一份 summary.json / summary.txt
out_dir = ROOT / "resnet18_baseline_summary"
out_dir.mkdir(parents=True, exist_ok=True)

(out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

summary_txt = []
summary_txt.append("PER-RUN TEST METRICS\n")
for r in rows:
    summary_txt.append(f"{r['dir']} (seed={r['seed']}): "
                       f"acc={r['acc']:.4f} precision={r['precision']:.4f} recall={r['recall']:.4f} "
                       f"f1={r['f1']:.4f} auc={r['auc']:.4f} cm={r['cm']}")
summary_txt.append("\nSUMMARY (mean ± std on TEST)\n")
for k in ["acc", "precision", "recall", "f1", "auc"]:
    m, s = mean_std(k)
    summary_txt.append(f"{k}: {m:.4f} ± {s:.4f}")

(out_dir / "summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

print(f"\nSaved: {out_dir / 'summary.txt'}")
print(f"Saved: {out_dir / 'summary.json'}")