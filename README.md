# 🚦 Trafik İşareti Tespit Projesi

## Proje Özeti
Bu projede, gerçek sürüş videolarından trafik işaretlerini tespit edip şekillerine göre sınıflandıracaksınız. Derin öğrenme kullanmadan, sadece klasik görüntü işleme teknikleri ile çözüm geliştireceksiniz.

---

## 📹 Veri Seti Kaynakları

### YouTube Dashcam Video Önerileri:
1. **"Driving in Germany - Autobahn A5"** - Almanya otoyol sürüşleri (çok sayıda işaret)
2. **"UK Dashcam - M25 London Orbital"** - İngiltere otoyol kayıtları
3. **"Japan Highway Driving 4K"** - Japonya otoyol sürüşleri
4. **"USA Interstate Highway Dashcam"** - Amerika eyaletler arası otoyol
5. **"European Roads Compilation"** - Avrupa yolları derlemesi

### Alternatif Hazır Veri Setleri:
- **GTSRB** (German Traffic Sign Recognition Benchmark)
- **Belgium Traffic Sign Dataset**
- **LISA Traffic Sign Dataset**

---

## 🎯 Tespit Edilecek İşaret Şekilleri

### 🔴 Daire/Yuvarlak İşaretler
- **Örnekler**: Hız limiti, park yasak, girilmez
- **Renk**: Genelde kırmızı çerçeve, beyaz iç
- **Tespit ipucu**: Hough Circle Transform

### 🔺 Üçgen İşaretler  
- **Örnekler**: Dikkat, yol ver, kaygan yol
- **Renk**: Kırmızı çerçeve, beyaz/sarı iç
- **Tespit ipucu**: Kontur analizi, köşe sayısı

### 🟦 Kare/Dikdörtgen İşaretler
- **Örnekler**: Otopark, hastane, bilgi levhaları
- **Renk**: Mavi zemin, beyaz sembol
- **Tespit ipucu**: Kontur analizi, aspect ratio

### 🛑 Sekizgen (STOP)
- **Örnekler**: Dur işareti
- **Renk**: Kırmızı zemin, beyaz yazı
- **Tespit ipucu**: 8 köşe tespiti

---

## 📁 Proje Klasör Yapısı

```
traffic_sign_detection/
│
├── input/
│   ├── video.mp4              # İndirilen dashcam videosu
│   └── test_images/            # Test için tek kareler
│
├── output/
│   ├── traffic_signs/          # Tespit edilen işaretler
│   │   ├── circular/           # Yuvarlak işaretler
│   │   ├── triangular/         # Üçgen işaretler
│   │   ├── rectangular/        # Kare/dikdörtgen işaretler
│   │   └── octagonal/          # Sekizgen işaretler
│   │
│   ├── detected_frames/        # İşaretli kareler
│   └── detection_log.csv       # Tespit kayıtları
│
├── src/
│   ├── video_processor.py      # Video işleme
│   ├── sign_detector.py        # İşaret tespiti
│   ├── shape_classifier.py     # Şekil sınıflandırma
│   └── utils.py                # Yardımcı fonksiyonlar
│
└── main.py                      # Ana program
```

---

## 🔧 Teknik Yaklaşım ve İpuçları

### 1️⃣ **Video İşleme Adımları**

```python
# Pseudo-kod yapısı
"""
1. Video'yu aç (cv2.VideoCapture)
2. Frame frame oku
3. Her frame için:
   - Ön işleme yap
   - İşaret tespit et
   - Şekil sınıflandır
   - Kaydet
4. Sonuçları raporla
"""
```

### 2️⃣ **Renk Segmentasyonu**

#### Kırmızı Renk Tespiti (HSV):
- **Alt sınır 1**: [0, 70, 50]
- **Üst sınır 1**: [10, 255, 255]
- **Alt sınır 2**: [170, 70, 50]  # Kırmızı HSV'de sarmalı
- **Üst sınır 2**: [180, 255, 255]

#### Mavi Renk Tespiti:
- **Alt sınır**: [100, 50, 50]
- **Üst sınır**: [130, 255, 255]

#### Sarı Renk Tespiti:
- **Alt sınır**: [20, 100, 100]
- **Üst sınır**: [30, 255, 255]

**💡 İpucu**: İki maske oluşturup OR işlemi yapın!

### 3️⃣ **Şekil Tespiti Algoritmaları**

#### A. Daire Tespiti:
```python
"""
Kullanılacak fonksiyonlar:
- cv2.HoughCircles()
- Parametreler: dp=1.2, minDist=30, param1=50, param2=30
- Dairesellik kontrolü: kontur alanı / (pi * r²)
"""
```

#### B. Üçgen Tespiti:
```python
"""
Adımlar:
1. Kontur bul (cv2.findContours)
2. Kontur yaklaşımı (cv2.approxPolyDP)
3. Köşe sayısı == 3 kontrolü
4. Alan filtresi (çok küçük/büyük olanları ele)
"""
```

#### C. Dikdörtgen Tespiti:
```python
"""
Kontroller:
1. 4 köşe
2. Aspect ratio: 0.8 < genişlik/yükseklik < 1.2
3. Diklik kontrolü (açılar ~90 derece)
"""
```

#### D. Sekizgen Tespiti:
```python
"""
STOP işareti için:
1. 8 köşe kontrolü
2. Kırmızı renk dominantlığı
3. Konvekslik kontrolü
"""
```

### 4️⃣ **Ön İşleme Pipeline'ı**

```python
"""
Önerilen sıralama:
1. Gaussian blur (gürültü azaltma)
2. Renk uzayı dönüşümü (BGR → HSV)
3. Renk maskeleme
4. Morfolojik işlemler (opening/closing)
5. Kenar tespiti (Canny)
6. Kontur bulma
7. Şekil analizi
"""
```

### 5️⃣ **ROI (Region of Interest) Belirleme**

```python
"""
Performans için:
- Görüntünün üst yarısına odaklan (gökyüzü)
- Yolun kenarlarına odaklan (işaretler genelde yanda)
- Çok uzak/yakın bölgeleri ihmal et
"""
```

---

## 📊 Tespit Kalitesi Kriterleri

### Filtreleme Parametreleri:
- **Minimum alan**: 500 piksel²
- **Maximum alan**: 50000 piksel²
- **Aspect ratio**: 0.5 - 2.0 arası
- **Solidity** (konvekslik): > 0.8
- **Minimum kontur noktası**: 5

### False Positive Azaltma:
1. Renk tutarlılığı kontrolü
2. Şekil simetrisi kontrolü
3. Çoklu frame doğrulama (tracking)
4. Non-maximum suppression

---

## 🔍 Debug ve Görselleştirme İpuçları

### Her Adımı Görselleştirin:
```python
"""
debug_mode = True ise:
- Orijinal frame
- HSV dönüşüm
- Renk maskesi
- Morfolojik sonuç
- Tespit edilen konturlar
- Final sonuç
"""
```

### Performans Metrikleri:
- FPS (Frame per second)
- Tespit edilen işaret sayısı
- İşleme süresi/frame
- Kayıp frame sayısı

---

## 💻 Kullanılacak OpenCV Fonksiyonları

### Temel Fonksiyonlar:
- `cv2.VideoCapture()` - Video okuma
- `cv2.cvtColor()` - Renk uzayı dönüşümü
- `cv2.inRange()` - Renk maskeleme
- `cv2.GaussianBlur()` - Bulanıklaştırma
- `cv2.Canny()` - Kenar tespiti

### Morfolojik İşlemler:
- `cv2.morphologyEx()` - Opening/Closing
- `cv2.erode()` - Erozyon
- `cv2.dilate()` - Dilatasyon

### Kontur İşlemleri:
- `cv2.findContours()` - Kontur bulma
- `cv2.approxPolyDP()` - Kontur yaklaşımı
- `cv2.contourArea()` - Alan hesaplama
- `cv2.boundingRect()` - Sınırlayıcı kutu
- `cv2.minEnclosingCircle()` - Minimum çember

### Şekil Tespiti:
- `cv2.HoughCircles()` - Daire tespiti
- `cv2.HoughLines()` - Çizgi tespiti
- `cv2.moments()` - Moment hesaplama

---

## 📝 Çıktı Format Önerileri

### Detection Log (CSV):
```csv
frame_no, timestamp, shape_type, x, y, width, height, confidence, color, saved_path
102, 00:00:03.4, circular, 234, 156, 45, 45, 0.89, red, output/circular/sign_001.jpg
105, 00:00:03.5, triangular, 456, 234, 60, 55, 0.92, red, output/triangular/sign_002.jpg
```

### Dosya İsimlendirme:
```
{shape_type}_{frame_number}_{timestamp}.jpg
Örnek: circular_0245_00_08_15.jpg
```

---

## 🎯 Bonus Görevler

1. **Hız Optimizasyonu**:
   - Multi-threading kullanımı
   - Frame skipping (her n. frame'i işle)
   - ROI ile işleme alanını küçült

2. **Tracking Ekleme**:
   - Tespit edilen işareti takip et

3. **İstatistik Raporu**:
   - En çok görülen işaret tipi
   - Zaman bazlı dağılım
   - Renk dağılımı analizi

4. **Video Overlay**:
   - Tespit edilen işaretleri video üzerine çiz
   - Bounding box ve etiket ekle
   - Yeni video olarak kaydet

5. **Adaptif Eşikleme**:
   - Işık koşullarına göre parametre ayarlama
   - Gece/gündüz tespiti

---

## ⚠️ Karşılaşılabilecek Zorluklar ve Çözümler

### Problem 1: Hareket Bulanıklığı
**Çözüm**: Frame kalitesi kontrolü, blur detection

### Problem 2: Değişken Işık Koşulları
**Çözüm**: Histogram eşitleme, adaptif threshold

### Problem 3: Perspektif Bozulması
**Çözüm**: Aspect ratio toleransı, kontur yaklaşımı

### Problem 4: Kısmi Görünüm
**Çözüm**: Minimum alan/kontur kontrolü

### Problem 5: Benzer Renkli Nesneler
**Çözüm**: Şekil doğrulama, çoklu kriter


---

## 📚 Faydalı Kaynaklar

1. **OpenCV Kontur Özellikleri**: 
   - Solidity, Aspect Ratio, Extent hesaplamaları

2. **Hough Transform Detaylı Açıklama**:
   - Circle detection parametreleri

3. **Color Space Dönüşümleri**:
   - RGB vs HSV avantajları

4. **Morphological Operations**:
   - Structuring element seçimi

---

**İyi çalışmalar! 🚗🚦**
