import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .model import build_model

def load_bundle(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(ckpt["model_name"], num_classes=ckpt["num_classes"], dropout=ckpt.get("dropout", 0.2), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v:k for k,v in label_to_id.items()}
    labels = [id_to_label[i] for i in range(len(id_to_label))]
    img_size = ckpt["img_size"]
    return model, labels, img_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    model, labels, img_size = load_bundle(Path(args.checkpoint))

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    top = torch.topk(probs, k=min(args.topk, len(labels)))
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{labels[idx]}: {p:.4f}")

if __name__ == "__main__":
    main()
