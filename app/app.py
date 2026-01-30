import json
from pathlib import Path

import torch
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

MODELS_DIR = Path(__file__).resolve().parent / "models"
CKPT_PATH = MODELS_DIR / "model.pt"
LABELS_PATH = MODELS_DIR / "labels.json"

def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model_name = ckpt.get("model_name", "efficientnet_b0")
    num_classes = ckpt["num_classes"]
    dropout = ckpt.get("dropout", 0.2)

    import timm
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, drop_rate=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # labels: prefer labels.json (from prepare_splits), else checkpoint mapping
    if LABELS_PATH.exists():
        meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        label_to_id = meta["label_to_id"]
        id_to_label = {int(i): lab for i, lab in meta["id_to_label"].items()} if "id_to_label" in meta else {v:k for k,v in label_to_id.items()}
        labels = [id_to_label[i] for i in range(len(id_to_label))]
    else:
        label_to_id = ckpt["label_to_id"]
        id_to_label = {v:k for k,v in label_to_id.items()}
        labels = [id_to_label[i] for i in range(len(id_to_label))]

    img_size = ckpt.get("img_size", 224)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, labels, tfm

MODEL, LABELS, TFM = None, None, None
if CKPT_PATH.exists():
    MODEL, LABELS, TFM = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    global MODEL, LABELS, TFM
    if request.method == "GET":
        return render_template("index.html", ready=MODEL is not None)

    if MODEL is None:
        return render_template("index.html", ready=False, error="Model bulunamadı. app/models/model.pt kopyalayın.")

    if "file" not in request.files:
        return render_template("index.html", ready=True, error="Dosya seçilmedi.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", ready=True, error="Dosya seçilmedi.")

    filename = secure_filename(file.filename)
    img = Image.open(file.stream).convert("RGB")
    x = TFM(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1).squeeze(0).tolist()

    ranked = sorted(list(zip(LABELS, probs)), key=lambda t: t[1], reverse=True)
    top1 = ranked[0]

    return render_template("index.html", ready=True, filename=filename, top1=top1, ranked=ranked)

if __name__ == "__main__":
    app.run(debug=True)
