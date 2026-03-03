import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def is_rgb_file(p: Path) -> bool:
    name = p.name.lower()
    if p.suffix.lower() not in IMG_EXTS:
        return False
    # 排除 test 里的派生图
    if "_depth_" in name or "_normal_" in name:
        return False
    # NeRF synthetic 的 RGB 通常是 r_*.png（train/val/test 都可能有）
    if name.startswith("r_") and p.suffix.lower() == ".png":
        return True
    # 有的版本 train/val 也可能只有 r_*.png，这里放宽：只要不是 depth/normal 就收
    return True

def list_rgb_images(cat_dir: Path, splits=("train", "val", "test")):
    files = []
    for sp in splits:
        sp_dir = cat_dir / sp
        if not sp_dir.exists():
            continue
        for f in sp_dir.rglob("*"):
            if f.is_file() and is_rgb_file(f):
                files.append(f)
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True, help="Root folder containing categories like lego/chair/...")
    parser.add_argument("--out_dir", type=str, default=r"E:\learn_pytorch\nerf\data_raw\nerf_pool")
    parser.add_argument("--per_class", type=int, default=300, help="Cap per category (to keep balance)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_dir = Path(args.out_dir)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    print("Found categories:", [c.name for c in cats])

    random.seed(args.seed)
    total_saved = 0

    for cat in cats:
        rgb_files = list_rgb_images(cat, splits=splits)
        if not rgb_files:
            print(f"[SKIP] {cat.name}: no rgb files found")
            continue

        random.shuffle(rgb_files)
        chosen = rgb_files[: min(args.per_class, len(rgb_files))]

        for i, f in enumerate(chosen):
            # 生成不会重复的名字：nerf_<cat>_<split>_<i>_<orig>
            split_name = f.parents[1].name  # .../<cat>/<split>/<file>  -> parents[1] == <split>
            new_name = f"nerf_{cat.name}_{split_name}_{i:04d}_{f.name}"
            shutil.copy2(f, out_dir / new_name)

        total_saved += len(chosen)
        print(f"[OK] {cat.name}: candidates={len(rgb_files)} saved={len(chosen)}")

    print(f"DONE. total_saved={total_saved} -> {out_dir}")

if __name__ == "__main__":
    main()