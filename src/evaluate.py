import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

from .data import DataTransforms, EyeDataset, read_split_csv
from .model import build_model
from .utils import ensure_dir, load_json, save_json, plot_confusion, compute_reports

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]
    num_classes = ckpt["num_classes"]
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v:k for k,v in label_to_id.items()}
    labels = [id_to_label[i] for i in range(len(id_to_label))]

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes, dropout=ckpt.get("dropout", 0.2), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    splits_dir = Path(args.splits_dir)
    test_paths, test_labels = read_split_csv(splits_dir/"test.csv")
    tfms = DataTransforms(img_size=img_size)
    ds = EyeDataset(test_paths, test_labels, label_to_id, tfms.eval_tfms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy().tolist())

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

    plots_dir = ensure_dir(out_dir/"plots")
    plot_confusion(y_true, y_pred, labels, plots_dir/"confusion_matrix.png")
    report = compute_reports(y_true, y_pred, labels)

    save_json({
        "accuracy": acc,
        "classification_report": report,
        "labels": labels,
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "img_size": img_size,
    }, out_dir/"eval.json")

    print("Saved:", out_dir)

if __name__ == "__main__":
    main()
