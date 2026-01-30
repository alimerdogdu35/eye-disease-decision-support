import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import IMG_EXTS

# Canonical class names (EN as dataset page)
CANONICAL = [
    "Retinitis Pigmentosa",
    "Retinal Detachment",
    "Pterygium",
    "Myopia",
    "Macular Scar",
    "Glaucoma",
    "Disc Edema",
    "Diabetic Retinopathy",
    "Central Serous Chorioretinopathy",
    "Healthy",
]

# Map common folder-name variants to canonical names.
# If your dataset uses different spelling, add it here.
ALIASES = {
    "retinitis pigmentosa": "Retinitis Pigmentosa",
    "rp": "Retinitis Pigmentosa",
    "retinal detachment": "Retinal Detachment",
    "retina dekolmani": "Retinal Detachment",
    "pterygium": "Pterygium",
    "pterjium": "Pterygium",
    "myopia": "Myopia",
    "miyopi": "Myopia",
    "macular scar": "Macular Scar",
    "makuler skar": "Macular Scar",
    "glaucoma": "Glaucoma",
    "glokom": "Glaucoma",
    "disc edema": "Disc Edema",
    "disk odemesi": "Disc Edema",
    "diabetic retinopathy": "Diabetic Retinopathy",
    "diyabetik retinopati": "Diabetic Retinopathy",
    "central serous chorioretinopathy": "Central Serous Chorioretinopathy",
    "santral seroz korioretinopati": "Central Serous Chorioretinopathy",
    "healthy": "Healthy",
    "normal": "Healthy",
    "saglikli": "Healthy",
}

def normalize_label(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, raw.strip())

def scan_images(data_dir: Path) -> List[Tuple[str, str]]:
    """Recursively scan for images; uses parent folder as label."""
    items: List[Tuple[str, str]] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label = normalize_label(p.parent.name)
            items.append((str(p), label))
    if not items:
        raise FileNotFoundError(f"No images found under: {data_dir}")
    return items

def read_split_csv(csv_path: Path) -> Tuple[List[str], List[str]]:
    paths, labels = [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            paths.append(row["path"])
            labels.append(row["label"])
    return paths, labels

@dataclass
class DataTransforms:
    img_size: int = 224

    def train_tfms(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def eval_tfms(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

class EyeDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[str], label_to_id: Dict[str,int], tfm):
        self.paths = paths
        self.labels = labels
        self.label_to_id = label_to_id
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        y = self.label_to_id[label]
        return x, y
