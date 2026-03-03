import argparse
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=r"E:\learn_pytorch\nerf\data_raw\real_ffhq")
    parser.add_argument("--n", type=int, default=1000)  # 500 或 1000
    parser.add_argument("--dataset", type=str, default="student/FFHQ")  # 一个可用的 HF 镜像
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    # streaming=True: 边读边取，不会把整个数据集下载到本地
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    saved = 0
    for ex in ds:
        if saved >= args.n:
            break

        # 尝试找到图片字段（不同镜像字段名可能不一样）
        img = None
        for key in ["image", "img", "png", "jpg"]:
            if key in ex:
                img = ex[key]
                break
        if img is None:
            continue

        # HF 的 image 一般是 PIL.Image
        if not isinstance(img, Image.Image):
            try:
                img = img.convert("RGB")
            except Exception:
                continue

        # 统一保存为 png，文件名可复现
        fp = out_dir / f"ffhq_{saved:06d}.png"
        img.convert("RGB").save(fp, format="PNG", optimize=True)

        saved += 1
        if saved % 50 == 0:
            print(f"Saved {saved}/{args.n} -> {out_dir}")

    print(f"DONE. Saved {saved} images to {out_dir}")

if __name__ == "__main__":
    main()