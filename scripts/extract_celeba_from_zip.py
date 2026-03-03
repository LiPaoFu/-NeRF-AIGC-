import argparse
import random
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, required=True, help="Path to img_align_celeba.zip")
    parser.add_argument("--dst", type=str, default=r"E:\learn_pytorch\nerf\data_split\test\real_celeba")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # 清空 dst（避免混）
    for p in dst.glob("*"):
        if p.is_file():
            p.unlink()

    with zipfile.ZipFile(zip_path, "r") as z:
        # zip 内部文件名列表（只要 jpg）
        jpgs = [name for name in z.namelist()
                if name.lower().endswith(".jpg") and not name.endswith("/")]
        if len(jpgs) < args.n:
            raise SystemExit(f"Zip has only {len(jpgs)} jpg files, need {args.n}")

        random.seed(args.seed)
        random.shuffle(jpgs)
        chosen = jpgs[:args.n]

        # 解压 chosen
        for i, name in enumerate(chosen):
            out = dst / f"celeba_{i:04d}.jpg"
            with z.open(name) as src, open(out, "wb") as f:
                f.write(src.read())

    (dst / "manifest.txt").write_text("\n".join(chosen), encoding="utf-8")
    print(f"DONE. Extracted {args.n} jpgs from {zip_path} -> {dst}")
    print(f"Saved manifest: {dst / 'manifest.txt'}")

if __name__ == "__main__":
    main()