import os
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

@dataclass
class CFG:
    data_dir: str = r"E:\learn_pytorch\nerf\data_split"
    out_dir: str = r"E:\learn_pytorch\nerf\runs\resnet18_baseline"
    model_dir: str = r"E:\learn_pytorch\nerf\models"

    img_size: int = 224
    batch_size: int = 8
    num_workers: int = 2  # Windows 上太大有时会卡，先用 2
    seed: int = 42

    # 训练策略：先冻结 backbone 再解冻微调（小数据更稳）
    epochs_head: int = 5
    epochs_finetune: int = 20
    lr_head: float = 3e-4
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 6  # val F1 连续不提升就停

    use_amp: bool = True  # 混合精度，省显存更稳

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs(cfg: CFG):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

def get_dataloaders(cfg: CFG):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.14)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(cfg.data_dir, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(cfg.data_dir, "test"),  transform=eval_tf)

    # 确保类名一致：ImageFolder 会按字母序
    # 你现在应该是 ["nerf", "real"]（因为 n < r）
    class_to_idx = train_ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, class_to_idx

def compute_class_weights(train_ds):
    # CrossEntropyLoss 的 weight: 每个类别一个权重，解决 120 vs 24 的不平衡
    labels = [y for _, y in train_ds.samples]
    counts = np.bincount(labels)
    # 权重 ~ 1/freq，再归一化到均值为1，数值更稳
    w = 1.0 / np.maximum(counts, 1)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts.tolist()

def build_model(num_classes=2):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

@torch.no_grad()
def evaluate(model, loader, device, amp, positive_class_index: int):
    model.eval()
    y_true = []
    y_prob = []
    y_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1)

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        y_prob.append(probs[:, positive_class_index].detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                       pos_label=positive_class_index)
    # ROC-AUC：当某个集合只有单一类别时会报错，做保护
    try:
        auc = roc_auc_score((y_true == positive_class_index).astype(int), y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "cm": cm.tolist()
    }

def train_one_phase(model, train_loader, val_loader, device, cfg: CFG,
                    criterion, optimizer, epochs: int, positive_class_index: int,
                    phase_name: str, scaler: torch.cuda.amp.GradScaler):

    best_f1 = -1.0
    best_path = None
    no_improve = 0

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"{phase_name} epoch {epoch}/{epochs}", ncols=100)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if cfg.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        val_metrics = evaluate(model, val_loader, device, cfg.use_amp, positive_class_index)
        train_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if k != "cm"}}
        history.append(row)

        print(f"\n[{phase_name}] epoch={epoch} train_loss={train_loss:.4f} "
              f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}")
        print(f"[{phase_name}] val_cm={val_metrics['cm']}")

        # 早停以 val F1 为准（小数据更靠谱）
        if val_metrics["f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["f1"]
            no_improve = 0
            best_path = os.path.join(cfg.model_dir, f"best_{phase_name}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "phase": phase_name,
                "epoch": epoch,
                "val_metrics": val_metrics
            }, best_path)
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                print(f"[{phase_name}] Early stopping triggered (patience={cfg.early_stop_patience}).")
                break

    return best_f1, best_path, history

def main():
    import argparse

    cfg = CFG()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only evaluate.")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path for eval_only mode.")
    args = parser.parse_args()

    cfg.seed = args.seed
    cfg.eval_only = args.eval_only
    cfg.ckpt = args.ckpt

    # 让每个 seed 输出到不同目录，避免 result.json 被覆盖
    cfg.out_dir = rf"E:\learn_pytorch\nerf\runs\resnet18_baseline_seed{cfg.seed}"

    set_seed(cfg.seed)
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(cfg)
    print("class_to_idx:", class_to_idx)

    # 正类定义：我们把 "nerf" 当作正类（更符合“检测伪造”）
    if "nerf" not in class_to_idx:
        raise RuntimeError(f"'nerf' class not found in ImageFolder classes: {class_to_idx}")
    positive_class_index = class_to_idx["nerf"]

    class_weights, counts = compute_class_weights(train_ds)
    print("train class counts:", counts, "weights:", class_weights.tolist())

    model = build_model(num_classes=len(class_to_idx)).to(device)

    if cfg.eval_only:
        if not cfg.ckpt:
            raise ValueError("--eval_only requires --ckpt <path_to_checkpoint>")

        print(f"[EVAL ONLY] Loading checkpoint: {cfg.ckpt}")
        ckpt = torch.load(cfg.ckpt, map_location=device)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)

        test_metrics = evaluate(model, test_loader, device, cfg.use_amp, positive_class_index)
        print("\n=== TEST METRICS ===")
        for k, v in test_metrics.items():
            if k != "cm":
                print(f"{k}: {v}")
        print("cm:", test_metrics["cm"])

        # 你之前加的 CelebA extra test（保持一致）
        celeba_test_root = os.path.join(cfg.data_dir, "test_celeba")
        if os.path.isdir(celeba_test_root):
            celeba_test_ds = datasets.ImageFolder(celeba_test_root, transform=test_ds.transform)
            celeba_test_loader = DataLoader(
                celeba_test_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True
            )
            celeba_metrics = evaluate(model, celeba_test_loader, device, cfg.use_amp, positive_class_index)
            print("\n=== EXTRA TEST METRICS (CelebA real) ===")
            for k, v in celeba_metrics.items():
                if k != "cm":
                    print(f"{k}: {v}")
            print("cm:", celeba_metrics["cm"])
        else:
            celeba_metrics = None
            print(f"[INFO] CelebA test root not found, skip: {celeba_test_root}")

        result = {
            "cfg": cfg.__dict__,
            "class_to_idx": class_to_idx,
            "positive_class_index": int(positive_class_index),
            "best_checkpoint": {"phase": "eval_only", "val_f1": None, "path": cfg.ckpt},
            "test_metrics": test_metrics,
            "extra_test_celeba_metrics": celeba_metrics
        }
        out_json = os.path.join(cfg.out_dir, "result_eval_only.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out_json}")
        return

    # 损失函数加权（解决 120 vs 24）
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # Phase 1: 只训练分类头
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)

    best_f1_head, best_path_head, hist_head = train_one_phase(
        model, train_loader, val_loader, device, cfg, criterion, optimizer,
        epochs=cfg.epochs_head, positive_class_index=positive_class_index,
        phase_name="head", scaler=scaler
    )

    # Phase 2: 解冻全模型微调
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

    best_f1_ft, best_path_ft, hist_ft = train_one_phase(
        model, train_loader, val_loader, device, cfg, criterion, optimizer,
        epochs=cfg.epochs_finetune, positive_class_index=positive_class_index,
        phase_name="finetune", scaler=scaler
    )

    # 选择更好的 checkpoint（head 或 finetune）
    candidates = []
    if best_path_head:
        candidates.append(("head", best_f1_head, best_path_head))
    if best_path_ft:
        candidates.append(("finetune", best_f1_ft, best_path_ft))
    best_phase, best_f1, best_ckpt = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    print(f"\nBEST checkpoint: phase={best_phase} val_f1={best_f1:.4f} path={best_ckpt}")

    # 加载 best 并在 test 上评估
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device, cfg.use_amp, positive_class_index)
    print("\n=== TEST METRICS ===")
    for k, v in test_metrics.items():
        if k != "cm":
            print(f"{k}: {v}")
    print("cm:", test_metrics["cm"])
    # === EXTRA TEST: CelebA real (cross-domain) ===
    celeba_test_root = os.path.join(cfg.data_dir, "test_celeba")
    if os.path.isdir(celeba_test_root):
        celeba_test_ds = datasets.ImageFolder(celeba_test_root, transform=test_ds.transform)

        if celeba_test_ds.class_to_idx != class_to_idx:
            print("[WARN] class_to_idx mismatch in CelebA test:",
                  celeba_test_ds.class_to_idx, "expected:", class_to_idx)

        celeba_test_loader = DataLoader(
            celeba_test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True
        )

        celeba_metrics = evaluate(model, celeba_test_loader, device, cfg.use_amp, positive_class_index)
        print("\n=== EXTRA TEST METRICS (CelebA real) ===")
        for k, v in celeba_metrics.items():
            if k != "cm":
                print(f"{k}: {v}")
        print("cm:", celeba_metrics["cm"])
    else:
        print(f"[INFO] CelebA test root not found, skip: {celeba_test_root}")

    # 保存一份结果到 runs/
    result = {
        "cfg": cfg.__dict__,
        "class_to_idx": class_to_idx,
        "positive_class_index": int(positive_class_index),
        "best_checkpoint": {"phase": best_phase, "val_f1": float(best_f1), "path": best_ckpt},
        "test_metrics": test_metrics
    }
    out_json = os.path.join(cfg.out_dir, "result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")

if __name__ == "__main__":
    main()