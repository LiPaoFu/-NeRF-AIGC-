import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="CelebA img_align_celeba folder")
    parser.add_argument("--dst", type=str, default=r"E:\learn_pytorch\nerf\data_split\test\real_celeba")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if len(files) < args.n:
        raise SystemExit(f"Not enough images in {src}. Found {len(files)}, need {args.n}")

    random.seed(args.seed)
    random.shuffle(files)
    chosen = files[:args.n]

    # 清空 dst 旧文件（避免混）
    for p in dst.glob("*"):
        if p.is_file():
            p.unlink()

    for i, f in enumerate(chosen):
        out = dst / f"celeba_{i:04d}{f.suffix.lower()}"
        shutil.copy2(f, out)

    (dst / "manifest.txt").write_text("\n".join(str(p) for p in chosen), encoding="utf-8")

    print(f"DONE. Copied {args.n} images -> {dst}")
    print(f"Saved manifest: {dst / 'manifest.txt'}")

if __name__ == "__main__":
    main()