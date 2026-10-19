# 🚀 MNIST El Yazısı Rakam Tanıma ve MLP Optimizasyon Analizi

Bu proje, MNIST el yazısı rakam tanıma veri seti kullanılarak, **Konvolüsyonel Sinir Ağları (CNN) veya hazır (pre-trained) modeller kullanılmadan**, sıfırdan bir Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) mimarisinin geliştirilmesi, adım adım optimize edilmesi ve laboratuvar verisi ile gerçek dünya verisi arasındaki **Dağılım Kayması (Distribution Shift)** olgusunun analiz edilmesi amacıyla hazırlanmıştır.

---

## 🎯 1. Beklenen Öğrenme Çıktılarının Karşılanması

Ödevin temel isterleri doğrultusunda aşağıdaki hedefler başarıyla gerçekleştirilmiştir:

* **MLP Mimarisi & Sıfırdan Eğitim:** CNN kullanılmadan tamamen `nn.Linear` katmanları ile inşa edilmiş, pre-trained hiçbir veri kullanılmamıştır.
* **Veri İzolasyonu:** Train, Validation ve Test split işlemleri manuel randomizasyon (Seed: 42) ile bölünmüş, `test_transform` ile Validation ve Test verilerine veri artırımı (augmentation) sızması engellenmiştir.
* **Hiperparametre Analizi:** Adam vs. SGD, CrossEntropy vs. MultiMarginLoss gibi farklı konfigürasyonların performansa etkileri grafiklerle kıyaslanmıştır.
* **Gerçek Dünya Verisi ile Genelleme (Generalization):** Öğrencinin kendi el yazısıyla oluşturduğu veri seti kullanılarak modelin "ezberleme" oranı test edilmiştir.

---

## 📂 2. Kendi Test Verimizi Oluşturma (Gerçek Dünya Verisi)

Modelin ezberleyip ezberlemediğini (overfitting) ölçmek için özel bir görüntü işleme hattı (pipeline) kurulmuştur. 

### Aşama 1: Ham Veri Toplama
Rakamlar A4 kağıdına kalemle yazılmış ve fotoğraflanmıştır. Modelin farklı varyasyonları görebilmesi için rakamlar farklı kalınlık ve eğimlerle yazılmıştır.

![Ham Veri - 6 Rakamları](ham_uretilen_test_data/WhatsApp%20Unknown%202026-10-16%20at%2014.49.11/alti_2.jpeg)
***Görsel 2.1: A4 kağıdına yazılmış ham gerçek dünya verisi (6 Rakamları)***

### Aşama 2: Otomatik Kırpma (`kirpma_2.py`)
Yazılan sayfalar OpenCV kullanılarak taranmış, `findContours` ile kontur tespiti yapılıp rakamlar ağırlık merkezine göre ortalanarak kare şeklinde kırpılmıştır.

![Kırpılmış Ham 6 Rakamı](kendi_test_verilerim/alti/alti_0cc276.jpg)
***Görsel 2.2: OpenCV algoritması ile otonom olarak bulunup kırpılan ham "6" rakamı***

### Aşama 3: Dijitalizasyon ve MNIST Formatı (`mnist_format_donusturme.py`)
Kırpılan resimler Otsu Thresholding ile tersine çevrilmiş (**siyah arka plan, beyaz rakam**) ve pürüzsüzleştirilerek 28x28 boyutuna indirgenmiştir. Hazırlanan veriler, yazılan özel `Dataset` sınıfı (`test_data_loader.py`) ile klasör isimlerinden otomatik etiketlenmiştir.

![MNIST Formatında 6 Rakamı](kendi_test_verilerim_mnist_format/alti/alti_01b011.jpg)
***Görsel 2.3: Tensör formatına girmeden önce 28x28 MNIST renk uzayına dönüştürülmüş final test verisi***

---

## 🔬 3. Model Evrimi ve Grafiksel Karşılaştırmalar

Modeller, kapasite genişletmenin etkisini net görmek adına Final (V7) modeline kadar **bilinçli olarak 16 nöronluk dar bir kapasitede (Bottleneck)** (`784 -> 16 -> 10`) tutulmuştur. 

### 🧠 Ağ Mimarisi Görselleştirmesi
Modelimizin girdi verisini (784 piksel) nasıl çok dar bir boğaza (16 nöron) sıkıştırıp ardından 10 sınıfa çıkardığını aşağıdaki sembolik mimari tasarımdan görebilirsiniz. Bu dar kapasite, modelin sınırlarını analiz etmemize ve yapılan optimizasyonların (loss, optimizer, scheduler vb.) etkisini net bir şekilde gözlemlememize olanak tanımıştır.

![MLP Mimarisi](Figure_1.png)
***Görsel 3.1: MNIST veri seti için tasarlanan 16 Nöronlu (Bottleneck) Çok Katmanlı Algılayıcı (MLP) Mimarisi***

### 📉 A. Temel Model (Base) ve Normalizasyonun Etkisi

Hiçbir işlem görmemiş `Base (v1)` modelimiz eğitimde iyi gibi görünse de, kendi yazdığım rakamlarda **%35.79**'da kalarak tamamen çökmüştür. Modele girdi verisini MNIST ortalamasıyla **Normalize** ettiğimizde ise başarı anında artmıştır.

![Base Model Grafiği](results/plots/mnist_base_v1_plot.png)
***Grafik 3.1: v1 Base Modeli Eğitim Metrikleri (Normalizasyon YOK - Gerçek Dünya Başarısı: %35.79)***

![Normalize Edilmiş Model](results/plots/mnist_normalized_v1_plot.png)
***Grafik 3.2: v1 Normalize Modeli Eğitim Metrikleri (Normalizasyon VAR - Gerçek Dünya Başarısı: %73.92)***

### 📉 B. Optimizatör Karşılaştırması: Adam vs SGD

Mimarimizi Adam yerine SGD optimizatörü ile eğittiğimizde yakınsamanın (convergence) ne kadar yavaş olduğu, Loss değerinin düşmekte zorlandığı aşağıdaki grafiklerde açıkça görülmektedir.

![Adam Optimizatör Grafiği](results/plots/mnist_normalized_v1_plot.png)
***Grafik 3.3: Adam Optimizatörü ile Hızlı Yakınsama (Test Accuracy: %95.38)***

![SGD Optimizatör Grafiği](results/plots/mnist_SGD_v2_plot.png)
***Grafik 3.4: SGD Optimizatörü ile Yavaş Yakınsama (Test Accuracy: %90.27)***

### 📉 C. Kayıp Fonksiyonu: CrossEntropy vs MultiMarginLoss

Sınıflandırmada geleneksel `CrossEntropy` yerine karar sınırlarını (margin) maksimize eden `MultiMarginLoss` denendiğinde modelin validasyon çizgisi (kırmızı çizgi) ciddi bir stabilite kazanmış ve gerçek dünya başarısı **%80.90**'a taşınmıştır.

![CrossEntropy Grafiği](results/plots/mnist_normalized_v1_plot.png)
***Grafik 3.5: CrossEntropy Loss Kullanımı (Validasyon çizgisindeki dalgalanmalara dikkat)***

![MultiMarginLoss Grafiği](results/plots/mnist_MultiMarginLoss_v3_plot.png)
***Grafik 3.6: MultiMargin Loss Kullanımı (Daha stabil ve pürüzsüz validasyon çizgisi)***

### 📉 D. Darboğazı Aşmak: Veri Artırımı ve LR Scheduler (V6)

Modelin 16 nöronluk kapasitesinin limitlerini görmek için uzamsal esneklik sağlayan `RandomAffine` ve lokal minimumlara pürüzsüz inmek için `StepLR` eklendi. Aşağıdaki grafikte StepLR'ın her 10 epoch'ta öğrenme oranını düşürmesiyle Loss'un nasıl kademeli olarak aşağı çekildiği net bir şekilde görülmektedir.

![V6 Scheduler Grafiği](results/plots/mnist_Scheduler_MultiMarginLoss_Deneme_v6_plot.png)
***Grafik 3.7: V6 Modeli - StepLR ve RandomAffine ile sınırların zorlanması***

### 📉 E. Final Modeli (V7): Kapasite Artışı ile Zirve Noktası

16 nöronluk limitler sonuna kadar "sömürüldükten" sonra, mimari genişletilerek `784 -> 64 -> 64 -> 10` formatına getirildi. V6'nın mükemmel kurgusuyla 40 epoch eğitilen V7 modeli, **MNIST testinde %98.58**, kendi el yazım verisinde ise **%96.30** başarı elde ederek Distribution Shift problemini tamamen çözmüştür.

![Final V7 Modeli Grafiği](results/plots/mnist_Scheduler_Nöron_sayısı_artırıldı_v7_plot.png)
***Grafik 3.8: V7 Final Modeli (64 Nöron) - Mükemmel Uyum ve Yüksek Genelleme Yeteneği***

#### ⚙️ V7 Mimari ve Hiperparametre Detayları
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

## 📊 4. Performans ve Metrik Karşılaştırma Tablosu

Tüm optimizasyonların "Laboratuvar Ortamı" (MNIST Test) ve "Gerçek Dünya" (Kendi El Yazım Testi) verisindeki başarımları aşağıdadır:

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

---

## 📁 5. Proje Dosya Yapısı

* `mnist.ipynb`: Veri setlerinin yüklendiği, tüm mimarilerin sıfırdan oluşturulup eğitildiği ana Jupyter Notebook dosyası.
* `kirpma_2.py`: Fiziksel A4 kağıtlarından el yazısı rakamları OpenCV ile tespit edip otonom kesen betik.
* `mnist_format_donusturme.py`: Kırpılan resimleri MNIST renk uzayına (Invert & Otsu Threshold) ve boyutuna (28x28) eşitleyen görüntü işleme kodu.
* `test_data_loader.py`: Özel fotoğrafları PyTorch formatına sokan ve klasör isimlerinden otomatik etiketleyen özel `Dataset` sınıfı.
* `toplu_test_2.py`: `results/models/` klasöründeki tüm eğitilmiş `.pth` modellerini çekip, öğrencinin kendi el yazısıyla toplu olarak test edip sonuçları `tum_modellerin_sonuclari.json` dosyasına yazdıran otomasyon kodu.

---

## 🏆 6. Sonuç ve Dağılım Kayması (Distribution Shift) Analizi

Bu proje kapsamında en çarpıcı bulgu, bir yapay zeka modelinin "Laboratuvar Ortamı" (MNIST) ile "Gerçek Dünya" (Kendi El Yazım) verisi arasındaki başarım farkının ne kadar dramatik olabileceğinin ispatlanmasıdır. Proje boyunca karşılaşılan temel sorunlar ve bu sorunlara getirilen mühendislik çözümleri aşağıda özetlenmiştir:

### ⚠️ A. Dağılım Kayması (Distribution Shift) Probleminin Gözlemi
* **Sorun:** Hiçbir ön işleme tabi tutulmayan ilk modelimiz (`Base V1`), standart MNIST test verisinde %94.72 doğruluk (accuracy) ile başarılı görünmesine rağmen, dış dünyadan (akıllı telefon kamerası) gelen ham verilerde **%35.79** gibi çok düşük bir başarı göstermiştir.
* **Analiz:** Bu durum klasik bir "Dağılım Kayması" örneğidir. Model, MNIST'in piksel dizilimlerini ve aydınlatma koşullarını "ezberlemiş" (Overfitting), ancak farklı bir kalem kalınlığı veya kağıt dokusu gördüğünde afallamıştır.

### 🛠️ B. Veri Sızıntısı (Data Leakage) Hatası ve Giderilmesi
* **Sorun (`V4` Modeli):** Modelin esnekliğini artırmak için eğitim verisine çok agresif dönüştürmeler (Data Augmentation) eklenmiş; ancak PyTorch `Dataset` yapısında `transform` fonksiyonu ayrıştırılmadığı için bu bozulmuş veriler **Validasyon Setine de sızmıştır.** Bu sızıntı sonucunda modelin MNIST başarım oranı aniden %73.06'ya çakılmıştır.
* **Çözüm (`V5` Modeli):** Sızıntıyı önlemek için eğitim ve test transformları kod seviyesinde birbirinden kesin olarak ayrılmıştır. Validasyon ve Test setleri her türlü sentetik dönüşümden (Rotation, Affine vb.) korunarak tamamen "temiz ve orijinal" bırakılmış; veri artırımı sadece eğitim (Train) setine kısıtlanmıştır. Bu sayede modelin öğrenme stabilitesi geri kazanılmıştır.

### 🧠 C. Optimizatör ve Kayıp Fonksiyonunun Adaptasyonu
* **Sorun:** Geleneksel `SGD` optimizatörünün oldukça yavaş yakınsadığı (convergence) ve klasik `CrossEntropyLoss` fonksiyonunun gerçek dünya verilerinde karar sınırlarını (decision boundaries) net çizemediği fark edilmiştir.
* **Çözüm:** Modelin öğrenme ivmesini artırmak için `Adam` optimizatörüne geçilmiş ve karar noktaları arasındaki mesafeyi (margin) maksimize etmeye odaklanan `MultiMarginLoss` kullanılmıştır. Bu değişiklik, özellikle kendi el yazım testlerinde başarıyı %73'ten %80.90'a zıplatmıştır.

### 📉 D. Lokal Minimumlardan Kaçış: Dinamik Öğrenme Oranı (StepLR)
* **Sorun:** 16 nöronluk dar bir darboğazda (Bottleneck) modelin ağırlık güncellemeleri zamanla durağanlaşmış ve "Optimal Çukur" noktasına inmekte zorlanmıştır.
* **Çözüm (`V6` Modeli):** Sabit bir öğrenme hızı (0.001) kullanmak yerine, PyTorch'un `StepLR` zamanlayıcısı (Scheduler) sisteme entegre edilmiştir. Her 10 epoch'ta bir öğrenme oranı `0.3` (Gamma) katsayısıyla küçültülerek, modelin en hassas hata noktalarına (Minimum Loss) pürüzsüzce yerleşmesi sağlanmış ve 16 nöronluk kapasiteden maksimum verim alınmıştır.

### 🚀 E. Nihai Çıkarım: Kapasite ve Genel Başarım
Sadece 16 nöron içeren dar bir MLP mimarisinde bile, **doğru veri ön işlemesi (OpenCV ile kırpma ve Normalizasyon), doğru hata fonksiyonu ve LR Scheduler** kullanıldığında başarımın %35'ten %88.56'ya çıkarılabileceği kanıtlanmıştır. Fiziksel limitlerin tükendiği bu noktada mimari `784 -> 64 -> 64 -> 10` olarak genişletildiğinde ise (`V7` Modeli), model gerçek hayat (Distribution Shift) sorununu tamamen aşarak kendi el yazım verisinde **%96.30** gibi muazzam bir doğruluğa ulaşmış ve ödevin tüm öğrenme çıktılarını başarıyla karşılamıştır.
