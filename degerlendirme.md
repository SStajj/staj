1) Alıştırma 1 – Değer Düzeltmesi

YANLIŞ (şu anki kod)

goruntu = np.array([...])  # 0 ve 1 değerleri kullanılmış


DOĞRU OLMASI GEREKEN

import numpy as np

goruntu = np.zeros((10, 10), dtype=np.uint8)

# Artı işareti için 255 değeri kullanılmalı
goruntu[5, :] = 255  # Yatay çizgi
goruntu[:, 4] = 255  # Dikey çizgi

2) Proje 4 – Panorama Birleştirme Sorunu

 Gerçek panorama görüntüleri ile test edilmeli (sentetik değil).

 dnm3.jpeg ve dnm4.jpeg yerine gerçek örtüşen fotoğraflar kullanılmalı.

 Kodda 'yuz3.png' referansı var ama yanlış dosya — düzeltilmeli.

 Feather blending bölümü test edilip düzgün çalıştığı doğrulanmalı.

3) Proje 5 – Input Kaldırılması

KALDIRILMASI GEREKEN

secim = input("Gürültü giderme yöntemi seçin...")
cekirdek_str = input("Kernel size giriniz...")


YERİNE

def main(yontem: str = 'gaussian', kernel_size: tuple[int, int] = (3, 3)):
    """Komut satırı etkileşimi yerine fonksiyon argümanları kullanın."""
    # İş mantığı burada...
    pass

🟡 Kod Kalitesi İyileştirmeleri
6) Hata Yönetimi Eklenmesi
import cv2
import numpy as np

def goruntu_yukle(dosya_yolu: str) -> np.ndarray | None:
    try:
        goruntu = cv2.imread(dosya_yolu)
        if goruntu is None:
            raise ValueError(f"Görüntü yüklenemedi: {dosya_yolu}")
        return goruntu
    except Exception as e:
        print(f"Hata: {e}")
        return None

7) İngilizce Değişken İsimleri

Şu anki: gurultusuz, maskelenmis, esitlenmis

Olması gereken: denoised, masked, equalized

🟢 Eklenmesi Gereken Dosyalar
8) requirements.txt Oluşturulması
numpy==1.24.3
opencv-python==4.8.0
matplotlib==3.7.1
Pillow==10.0.0

12) Görüntü Dosyaları Kontrolü

 lena.jpg tüm klasörlerde tekrarlanmış — tek bir data/images/ klasöründe tutulmalı.

 Gereksiz görüntü dosyaları temizlenmeli.

Önerilen yapı

data/
  images/
    lena.jpg
    pano_01.jpg
    pano_02.jpg
src/
  ...
README.md
requirements.txt

🔧 Performans İyileştirmeleri
13) Vektörizasyon Kullanımı

YAVAŞ (Alıştırma 6'da)

for i in range(yukseklik):
    for j in range(genislik):
        # işlem


HIZLI: NumPy vektörizasyon kullan

# Örnek: eşiğin üstündeki pikselleri beyaz yap
mask = image > threshold
result = np.where(mask, 255, 0).astype(np.uint8)


Not: np.vectorize kullanımı API’yi sadeleştirir ama çoğu durumda gerçek hız kazancı sağlamaz. Mümkün olduğunca ufunc, boolean indeksleme ve yayılım (broadcasting) tercih edin.

14) Bellek (Memory) Optimizasyonu
# dtype belirtilmeli: uint8 çoğu görüntü için yeterlidir
image = np.zeros((height, width), dtype=np.uint8)

# Yüzer sayılar gerektiğinde float32 tercih edin
arr = arr.astype(np.float32, copy=False)  # float64 yerine float32
