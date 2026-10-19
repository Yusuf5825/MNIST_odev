# MNIST Multi-Layer Perceptron (MLP) Optimizasyon ve Gerçek Dünya Verisi Analizi

Bu proje, MNIST el yazısı rakam tanıma veri seti kullanılarak, Konvolüsyonel Sinir Ağları (CNN) veya hazır (pre-trained) modeller **kullanılmadan**, sıfırdan bir Çok Katmanlı Algılayıcı (MLP) mimarisinin geliştirilmesi, sınırlarının zorlanması ve gerçek dünya verilerindeki başarımının test edilmesi amacıyla hazırlanmıştır.

## 🎯 Proje Amacı ve Kapsamı
1. Temel bir MLP mimarisinin hiperparametre optimizasyonu, kayıp fonksiyonu (loss function) analizleri, veri artırımı (data augmentation) ve öğrenme oranı zamanlayıcıları (LR Scheduler) ile adım adım iyileştirilmesi.
2. Eğitim, Doğrulama (Validation) ve Test setlerinin doğru bir şekilde izole edilerek veri sızıntısının (Data Leakage) önlenmesi.
3. Modelin sadece MNIST veri setinde değil, **öğrenci tarafından fiziksel A4 kağıdına kalemle yazılıp OpenCV ile dijitalize edilen kendi el yazısı veri setinde** test edilerek laboratuvar verisi ile gerçek dünya verisi arasındaki "Distribution Shift" (Dağılım Kayması) olgusunun incelenmesi.

---

## 📂 1. Kullanılan Veri Setleri ve Veri Hazırlığı

### A. Orijinal MNIST Veri Seti
Model eğitiminde 28x28 boyutlarında, 10 sınıftan (0-9) oluşan standart MNIST veri seti kullanılmıştır. `torch.utils.data.random_split` ile 60.000'lik ana veri; **50.000 Train** ve **10.000 Validation** olarak bölünmüş, test verisi tamamen yalıtılmıştır.

### B. Kendi El Yazım (Gerçek Dünya Verisi)
Modelin gerçek dünya genelleme (generalization) yeteneğini ölçmek için özel bir veri seti üretilmiştir:
1. **Oluşturma:** Rakamlar A4 kağıdına kalın uçlu kalemle yazılmış ve fotoğraflanmıştır.
2. **Otomasyon (`kirpma.py`):** Yazılan sayfalar OpenCV kullanılarak taranmış, kontur tespiti (`findContours`) ile her bir rakam bulunup, ağırlık merkezine göre ortalanmış (Center of Mass) ve tam kare şeklinde kırpılarak UUID ile isimlendirilip sınıflarına göre klasörlenmiştir.
3. **MNIST Formatına Dönüşüm (`mnist_format_donusturme.py`):** Kırpılan resimler gri tonlamaya çevrilmiş, Otsu Thresholding ile renkler tersine çevrilmiş (siyah arka plan, beyaz rakam) ve *Anti-aliasing* (`INTER_AREA`) ile pürüzsüzleştirilerek 28x28 boyutuna indirgenmiştir.
4. **Otonom Yükleme (`test_data_loader.py`):** Hazırlanan veriler yazılan özel PyTorch `Dataset` sınıfı ile klasör isimlerinden otonom olarak etiketlenmiş ve modele sunulmuştur.

---

## 🚀 2. Deney Süreci ve Model Evrimi (Bottleneck Analizi)

Modeller, kapasite genişletmenin etkisini net görmek adına V7'ye kadar **bilinçli olarak 16 nöronluk dar bir kapasitede (Bottleneck)** (`784 -> 16 -> 10`) tutulmuş, fiziksel sınırlara ulaşıldığında V7'de 64 nörona çıkılmıştır.

* **Faz 1 - Temel Kurulum (`v1`, `v2`, `v3`):** 
  * Hiçbir işlem görmemiş `v1` modeli MNIST testinde %94.72 başarım gösterse de, gerçek dünya verisinde **%35.79**'da kalarak ezberlediğini (overfitting) göstermiştir.
  * Veriler MNIST istatistikleriyle (`mean=0.1307, std=0.3081`) normalize edilmiş ve `Adam` optimizatörünün `SGD`'den çok daha hızlı yakınsadığı görülmüştür.
  * Sınıflandırmada `CrossEntropyLoss` yerine, karar sınırlarını maksimize eden `MultiMarginLoss` denenmiş ve başarı artmıştır.
* **Faz 2 - Veri Artırımı ve Sızıntı (Data Leakage) Çözümü (`v4`, `v5`):** 
  * `v4` modelinde `TrivialAugmentWide` gibi agresif dönüşümler uygulanmış, ancak Train/Test Transformları ayrılmadığı için Validasyon setine de artırılmış veri gitmiş ve başarım %73'e çakılmıştır.
  * `v5` modellerinde sızıntı giderilmiş, sadece Eğitim setine `RandomRotation` uygulanarak %95+ başarılara geri dönülmüştür.
* **Faz 3 - Kapasiteyi Sömürme (`v6`):**
  * MLP'nin uzamsal zayıflığını gidermek için `RandomAffine`, lokal minimumlara pürüzsüz inmek için `StepLR` (Her 10 epoch'ta gamma=0.3) ve `L2 Regularization` (weight_decay=1e-4) eklenerek 40 epoch eğitilmiştir. Dar kapasitenin limitlerine ulaşıldığı kanıtlanmıştır.
* **Faz 4 - Kapasite Genişletme (`v7`):**
  * Mimari `784 -> 64 -> 10` olarak güncellenmiş ve V6'nın mükemmel kurgusuyla eğitilmiştir. Model gerçek dünya verisinde **%96.30** başarıya ulaşarak genelleme problemini çözmüştür.

---

## 📊 3. Performans Karşılaştırma Tablosu

Aşağıdaki tablo, model optimizasyonlarının hem standart MNIST test verisindeki hem de tamamen farklı dağılıma sahip kendi el yazısı (Distribution Shift) verisindeki başarımlarını özetlemektedir:

| Model Adı | Gizli Nöron | Optimizatör & Loss | Özellik / İyileştirme | MNIST Test Accuracy | Kendi El Yazım Accuracy |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **v1 (Base)** | 16 | Adam, CrossEntropy | Ham Veri, Transform Yok | %94.72 | **%35.79** |
| **v1 (Normalized)**| 16 | Adam, CrossEntropy | Sadece Normalizasyon | %95.38 | %73.92 |
| **v2 (SGD)** | 16 | SGD, CrossEntropy | Adam yerine SGD | %90.27 | %71.44 |
| **v3 (MultiMargin)**| 16 | Adam, MultiMargin | Hata fonksiyonu değişti | %95.39 | %80.90 |
| **v4 (Aug Leakage)**| 16 | Adam, CrossEntropy | Aşırı Augment + Validasyon Sızıntısı | %73.06 | %77.21 |
| **v5 (Minimize Aug)**| 16 | Adam, CrossEntropy | Sızıntı Çözüldü + RandomRotation | %95.14 | %80.79 |
| **v5 (MM + Aug)** | 16 | Adam, MultiMargin | V5'e MultiMargin Eklendi | %95.30 | %82.61 |
| **v6 (Scheduler)** | 16 | Adam, MultiMargin | RandomAffine + StepLR + L2 Reg. (30 Ep) | %93.42 | %86.71 |
| **v6 (Scheduler+)** | 16 | Adam, MultiMargin | Gamma 0.3 + (40 Ep) Limitsel Zirve | %93.63 | %88.56 |
| **v7 (Final Model)**| **64** | **Adam, MultiMargin** | **V6 Ayarları + Kapasite Artışı** | **%98.58** | **%96.30** |

> *Çıkarım: Tablodan da görüleceği üzere, Base model laboratuvar ortamında (MNIST Test) %94 başarı gösterse de gerçek hayatta (Kendi El Yazım) %35 ile çökmüştür. Veri artırımı ve mimari optimizasyonlar sayesinde gerçek hayat başarısı adım adım %96.3'e çıkarılarak "Distribution Shift" sorunu aşılmıştır.*

---

## 📈 4. Nihai Model (V7) Eğitim Grafikleri

Aşağıdaki grafikler `mnist_Scheduler_Nöron_sayısı_artırıldı_v7` modelinin 40 epoch boyunca Loss ve Accuracy değişimlerini göstermektedir. 

![V7 Eğitim Grafikleri](results/plots/mnist_Scheduler_Nöron_sayısı_artırıldı_v7_plot.png)
*(Not: `StepLR` kullanıldığı için her 10 epoch'ta bir Learning Rate düşüşüne bağlı olarak validasyon kaybındaki pürüzsüzleşme net bir şekilde gözlemlenmektedir.)*

### ⚙️ V7 Mimari ve Hiperparametre Detayları
```python
current_hyperparameters = {
    "epochs": 40,
    "batch_size": 32,
    "initial_learning_rate": 0.001,
    "scheduler": "StepLR(step_size=10, gamma=0.3)",
    "optimizer": "Adam (weight_decay=1e-4)",
    "loss_function": "MultiMarginLoss()",
    "transform": "RandomAffine(degrees=[-10.0, 10.0], translate=(0.1, 0.1), scale=(0.9, 1.1))",
    "architecture": "MLP (Input: 784 -> Hidden: 64 + ReLU -> Hidden: 64 + ReLU -> Output: 10)"
}
```

---

## 📁 5. Proje Dosya Yapısı
* `mnist.ipynb`: Veri setlerinin yüklendiği, tüm mimarilerin sıfırdan oluşturulup eğitildiği ana Jupyter Notebook dosyası.
* `kirpma.py`: Fiziksel A4 kağıtlarından el yazısı rakamları OpenCV ile tespit edip otonom kesen betik.
* `mnist_format_donusturme.py`: Kırpılan resimleri MNIST renk uzayına (Invert & Otsu Threshold) ve boyutuna (28x28) eşitleyen görüntü işleme kodu.
* `test_data_loader.py`: Özel fotoğrafları PyTorch formatına sokan ve klasör isimlerinden otomatik etiketleyen özel `Dataset` sınıfı.
* `toplu_test.py`: `models/` klasöründeki tüm eğitilmiş `.pth` modellerini çekip, öğrencinin kendi el yazısıyla toplu olarak test edip sonuçları `tum_modellerin_sonuclari.json` dosyasına yazdıran otomasyon kodu.

---

## 🎓 6. Beklenen Öğrenme Çıktılarının Karşılanması
* **MLP Mimarisi & Sıfırdan Eğitim:** CNN kullanılmadan tamamen `nn.Linear` katmanları ile inşa edilmiş, pre-trained hiçbir veri kullanılmamıştır.
* **Veri İzolasyonu:** Train, Validation ve Test split işlemleri manuel randomizasyon (Seed: 42) ile bölünmüş, `test_transform` ile Validation ve Test verilerine veri artırımı (augmentation) sızması engellenmiştir.
* **Hyperparameter Analizi:** Tablodaki gelişim, Loss ve Optimizatör değişikliklerinin (SGD vs Adam, CrossEntropy vs MultiMargin) etkisini kanıtlamaktadır.
* **Generalization (Genelleme) Gözlemi:** Özel el yazısı testi betiği ile modelin sadece gördüğü veri tipini ezberlemediği, farklı kalem ve kağıt tipleriyle yazılmış, OpenCV ile taranmış verileri de başarıyla tahmin edebildiği ispatlanmıştır.