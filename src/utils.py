import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def plot_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    # history keys: train_loss, val_loss, train_acc, val_acc
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(list(epochs), history["train_loss"])
    plt.plot(list(epochs), history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Eğitim / Doğrulama Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(list(epochs), history["train_acc"])
    plt.plot(list(epochs), history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Eğitim / Doğrulama Accuracy")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(out_path.with_name("acc_curves.png"), dpi=200)
    plt.close()

def plot_confusion(y_true: List[int], y_pred: List[int], labels: List[str], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def compute_reports(y_true: List[int], y_pred: List[int], labels: List[str]) -> Dict:
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    return report
