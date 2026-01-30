# Eye Disease Decision Support System (CNN/DL + Flask)

Bu repo, **Mendeley Data - Eye Disease Image Dataset (DOI: 10.17632/s9bfhswzjb.1)** ile
**10 sınıflı** göz hastalığı sınıflandırma modeli eğitmek ve eğitilen modeli **Flask**
tabanlı bir web arayüzü ile “karar destek sistemi” olarak sunmak için hazırlanmıştır.

## 1) Kapsam
- Eğitim: Transfer Learning (EfficientNet-B0 / ResNet50 seçilebilir) + PyTorch
- Çıktılar: `outputs/` altında model ağırlıkları, metrikler, eğitim grafiklerini ve confusion matrix görsellerini üretir
- Arayüz: `app/` altında Flask ile resim yükle -> sınıf tahmini + olasılıklar

## 2) Kurulum
> Python 3.10+ önerilir

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 3) Veri setini indirme ve hazırlama
1. Mendeley sayfasından veri setini indirip bir klasöre çıkartın (zip).
2. Çıkartılan klasör yolunu `DATA_DIR` olarak vereceğiz.

Repo, sınıf klasörlerini **otomatik tarar**. Örnek:
```
DATA_DIR/
  Augmented/
    Glaucoma/
    Healthy/
    ...
  Original/
    ...
```
Klasör yapınız farklıysa sorun değil; `src/data.py` içindeki tarayıcı **alt klasörlerdeki**
resimleri bulur ve **klasörü etiket** olarak kullanır.

### Split (train/val/test) üretimi
```bash
python -m src.prepare_splits --data_dir "PATH/TO/DATA_DIR" --out_dir outputs/splits --seed 42
```

Bu komut:
- `train.csv`, `val.csv`, `test.csv` üretir (path + label)
- sınıf isimlerini normalize eder (`labels.json`)

## 4) Model eğitimi
```bash
python -m src.train   --splits_dir outputs/splits   --model efficientnet_b0   --img_size 224   --batch_size 32   --epochs 20   --lr 3e-4   --num_workers 4   --out_dir outputs/run1
```

Eğitim sonunda:
- `outputs/run1/best_model.pt` (en iyi doğrulama modeli)
- `outputs/run1/metrics.json`
- `outputs/run1/plots/` (loss/acc eğrileri, confusion matrix, PR/ROC gibi görseller)

## 5) Değerlendirme
```bash
python -m src.evaluate --checkpoint outputs/run1/best_model.pt --splits_dir outputs/splits --out_dir outputs/run1_eval
```

## 6) Flask karar destek sistemi
Modeli kopyalayın:
```bash
mkdir -p app/models
cp outputs/run1/best_model.pt app/models/model.pt
cp outputs/splits/labels.json app/models/labels.json
```

Arayüzü çalıştırın:
```bash
export FLASK_APP=app/app.py
export FLASK_ENV=development
flask run --host 0.0.0.0 --port 5000
```

Tarayıcı: `http://localhost:5000`


## 7) Lisans / Atıf
Veri seti lisansı CC BY 4.0 (Mendeley).

---

> Not: Bu repo, değerlendirme için beklenen **“çalıştırılabilir kaynak kod + rapor şablonu”** sağlar.
> Model performans değerleri; sizin makinenizde veri seti ile eğitiminiz sonrası otomatik üretilecektir.
