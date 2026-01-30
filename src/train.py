import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import DataTransforms, EyeDataset, read_split_csv
from .model import build_model
from .utils import set_seed, ensure_dir, load_json, save_json, plot_curves

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="efficientnet_b0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = ensure_dir(Path(args.out_dir))
    splits_dir = Path(args.splits_dir)

    labels_meta = load_json(splits_dir/"labels.json")
    label_to_id = labels_meta["label_to_id"]
    id_to_label = {int(k):v for k,v in labels_meta["id_to_label"].items()}
    num_classes = len(label_to_id)

    train_paths, train_labels = read_split_csv(splits_dir/"train.csv")
    val_paths, val_labels = read_split_csv(splits_dir/"val.csv")

    tfms = DataTransforms(img_size=args.img_size)
    train_ds = EyeDataset(train_paths, train_labels, label_to_id, tfms.train_tfms())
    val_ds = EyeDataset(val_paths, val_labels, label_to_id, tfms.eval_tfms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=num_classes, dropout=args.dropout, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device=device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"[{epoch:03d}/{args.epochs:03d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        # Save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "label_to_id": label_to_id,
                "img_size": args.img_size,
                "dropout": args.dropout,
            }, out_dir/"best_model.pt")

    elapsed = time.time() - t0

    # Plots + metrics
    plots_dir = ensure_dir(out_dir/"plots")
    plot_curves(history, plots_dir/"loss_curves.png")

    save_json({
        "model": args.model,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_val_acc": best_val_acc,
        "history": history,
        "elapsed_sec": elapsed,
    }, out_dir/"metrics.json")

    print("Saved:", out_dir/"best_model.pt")
    print("Metrics:", out_dir/"metrics.json")

if __name__ == "__main__":
    main()
