import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def copy_images(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            shutil.copy2(p, dst / p.name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_test_root", type=str, default=r"E:\learn_pytorch\nerf\data_split\test")
    parser.add_argument("--celeba_real_dir", type=str, default=r"E:\learn_pytorch\nerf\data_split\test\real_celeba")
    parser.add_argument("--out_test_root", type=str, default=r"E:\learn_pytorch\nerf\data_split\test_celeba")
    args = parser.parse_args()

    orig_test_root = Path(args.orig_test_root)
    celeba_real_dir = Path(args.celeba_real_dir)
    out_test_root = Path(args.out_test_root)

    # clean
    if out_test_root.exists():
        shutil.rmtree(out_test_root)
    (out_test_root / "nerf").mkdir(parents=True, exist_ok=True)
    (out_test_root / "real").mkdir(parents=True, exist_ok=True)

    # copy nerf from original test
    copy_images(orig_test_root / "nerf", out_test_root / "nerf")

    # copy real from celeba_real_dir
    copy_images(celeba_real_dir, out_test_root / "real")

    nerf_n = len([p for p in (out_test_root / "nerf").iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    real_n = len([p for p in (out_test_root / "real").iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    print(f"DONE. Built ImageFolder test root: {out_test_root}")
    print(f"Counts: nerf={nerf_n}, real={real_n}")

if __name__ == "__main__":
    main()