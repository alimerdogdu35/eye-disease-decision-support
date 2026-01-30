# Göz Hastalıkları Karar Destek Sistemi (CNN/DL)

## 1. Giriş
Göz hastalıkları, yaşam kalitesini doğrudan etkileyen ve geç kalındığında kalıcı görme kaybına kadar ilerleyebilen önemli sağlık problemleridir.
Özellikle retina kaynaklı hastalıklar (Diyabetik Retinopati, Retina Dekolmanı vb.) erken aşamada tespit edilirse tedavi başarısı artmaktadır.

Bu projede, fundus kamera görüntülerinden oluşan **Eye Disease Image Dataset** kullanılarak 10 sınıflı bir sınıflandırma modeli eğitilmiş
ve eğitilen model **Flask tabanlı** bir web arayüzü ile karar destek sistemi haline getirilmiştir.

**Veri seti**: Mendeley Data üzerinde yayınlanan “Eye Disease Image Dataset” (DOI: 10.17632/s9bfhswzjb.1).
Veri seti; Retinitis Pigmentosa, Retinal Detachment, Pterygium, Myopia, Macular Scar, Glaucoma, Disc Edema,
Diabetic Retinopathy, Central Serous Chorioretinopathy ve Healthy sınıflarını içerir.

## 2. Yöntem

### 2.1. Ön İşleme ve Veri Bölme
- Görüntüler RGB formatına çevrilir ve `224x224` boyutuna ölçeklenir.
- Eğitim sırasında veri artırma (augmentation):
  - RandomHorizontalFlip
  - RandomRotation (±10 derece)
  - ColorJitter (parlaklık/kontrast/doygunluk küçük değişimler)
- Veri seti stratified olarak **train/val/test** bölünür (`prepare_splits.py`).

### 2.2. Model Mimarisi
Bu çalışmada Transfer Learning yaklaşımı kullanılmıştır.

Örnek konfigürasyon:
- Backbone: `efficientnet_b0` (timm)
- Çıkış katmanı: 10 sınıf (softmax)
- Dropout: 0.2

> Not: Aynı altyapı ile `resnet50` gibi farklı CNN omurgaları da denenebilir.

### 2.3. Kayıp Fonksiyonu ve Optimizasyon
- Loss: CrossEntropyLoss (opsiyonel label smoothing)
- Optimizer: AdamW
- Öğrenme oranı: 3e-4
- Weight decay: 1e-4
- Epoch: 20 (örnek)

## 3. Sonuçlar

Aşağıdaki çıktılar eğitim sonrası otomatik üretilir:
- Accuracy/Loss eğrileri: `outputs/<run>/plots/loss_curves.png`, `acc_curves.png`
- Confusion Matrix: `outputs/<eval>/plots/confusion_matrix.png`
- Sınıf bazlı precision/recall/f1: `outputs/<eval>/eval.json`

### 3.1. Değerlendirme Metrikleri (Test Set)
Buraya `outputs/run1_eval/eval.json` içindeki sonuçlar rapora aktarılmalıdır.

Örnek tablo şablonu:

| Metrik | Değer |
|---|---:|
| Accuracy | ... |
| Macro F1 | ... |
| Weighted F1 | ... |

### 3.2. Eğitim Grafik Görselleri
Aşağıdaki görseller rapora eklenmelidir:
- Eğitim / Doğrulama Loss
- Eğitim / Doğrulama Accuracy
- Confusion Matrix

## 4. Karar Destek Sistemi Arayüzü
Flask arayüzünde kullanıcı fundus görüntüsü yükler; model:
- en yüksek olasılıklı sınıfı,
- tüm sınıfların olasılık dağılımını
ekranda gösterir.

## 5. Sonuç ve Gelecek Çalışmalar
- Daha güçlü omurgalar (EfficientNetV2, ConvNeXt)
- Sınıf dengesizliğine karşı Focal Loss / class weights
- Grad-CAM ile açıklanabilirlik
- Modelin mobil/edge dağıtımı (TorchScript/ONNX)

## 6. Kaynaklar
- Mendeley Data: Eye Disease Image Dataset (DOI: 10.17632/s9bfhswzjb.1)
- PyTorch, timm, Flask dokümantasyonları
