import random
import shutil
from pathlib import Path

SEED = 42
SPLIT = (0.7, 0.15, 0.15)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_NERF = PROJECT_ROOT / "data_raw" / "nerf_pool"
RAW_REAL = PROJECT_ROOT / "data_raw" / "real_ffhq"
OUT = PROJECT_ROOT / "data_split"
# =========================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def list_images(p: Path):
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_files(files, dst_dir: Path, prefix: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        # 关键：输出文件名必须唯一，避免 lego/chair/drums 同名覆盖
        # f.parent.name 会是 lego/chair/drums 或 real_ffhq
        new_name = f"{prefix}_{f.parent.name}_{f.name}"
        shutil.copy2(f, dst_dir / new_name)

def split_list(items, split):
    n = len(items)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

def main():
    random.seed(SEED)

    # 1) 收集 real
    real_files = list_images(RAW_REAL)
    random.shuffle(real_files)

    # 2) 收集 nerf（把 lego/chair/drums 合并为一个“nerf”类，但先在每个类别内部打散再合并，避免偏差）
    nerf_files = list_images(RAW_NERF)
    random.shuffle(nerf_files)

    # 3) 打散后划分
    random.shuffle(nerf_files)

    real_train, real_val, real_test = split_list(real_files, SPLIT)
    nerf_train, nerf_val, nerf_test = split_list(nerf_files, SPLIT)

    # 4) 清空旧输出（可选：稳妥起见每次重建）
    if OUT.exists():
        shutil.rmtree(OUT)
    # 5) 创建目录
    for split_name in ["train", "val", "test"]:
        safe_mkdir(OUT / split_name / "real")
        safe_mkdir(OUT / split_name / "nerf")

    # 6) 复制
    copy_files(real_train, OUT / "train" / "real", prefix="real")
    copy_files(real_val, OUT / "val" / "real", prefix="real")
    copy_files(real_test, OUT / "test" / "real", prefix="real")

    copy_files(nerf_train, OUT / "train" / "nerf", prefix="nerf")
    copy_files(nerf_val, OUT / "val" / "nerf", prefix="nerf")
    copy_files(nerf_test, OUT / "test" / "nerf", prefix="nerf")

    print("=== DONE ===")
    print(f"Real:  train={len(real_train)} val={len(real_val)} test={len(real_test)} total={len(real_files)}")
    print(f"Nerf:  train={len(nerf_train)} val={len(nerf_val)} test={len(nerf_test)} total={len(nerf_files)}")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()