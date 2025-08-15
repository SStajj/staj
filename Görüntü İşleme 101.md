# Görüntü İşleme 101: Temellerden Uygulamaya

## Giriş

Görüntü işleme, dijital görüntüler üzerinde matematiksel ve algoritmik işlemler yaparak bilgi çıkarma, görüntüyü iyileştirme veya dönüştürme sanatıdır. Bu dokümanda, görüntü işlemenin temellerini öğrenecek ve Python ile sıfırdan kodlayarak uygulayacaksınız.

### Gerekli Kütüphaneler
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
```

---

## Bölüm 1: Dijital Görüntü Nedir?

### 1.1 Piksel ve Görüntü Matrisi

Dijital görüntü, aslında sayılardan oluşan bir matristir. Her bir sayı, görüntünün en küçük birimi olan **piksel**'i temsil eder.

- **Gri tonlamalı görüntü**: 2 boyutlu matris (MxN)
- **Renkli görüntü**: 3 boyutlu matris (MxNx3)

```python
# Basit bir gri tonlamalı görüntü oluşturalım
import numpy as np
import matplotlib.pyplot as plt

# 5x5'lik basit bir görüntü matrisi
goruntu = np.array([
    [0, 50, 100, 150, 200],
    [50, 100, 150, 200, 250],
    [100, 150, 200, 250, 255],
    [150, 200, 250, 255, 200],
    [200, 250, 255, 200, 150]
])

plt.imshow(goruntu, cmap='gray')
plt.colorbar()
plt.title('5x5 Piksel Görüntü')
plt.show()

print(f"Görüntü boyutu: {goruntu.shape}")
print(f"(2,3) konumundaki piksel değeri: {goruntu[2,3]}")
```

### 📝 Alıştırma 1: Kendi Görüntünüzü Oluşturun
10x10'luk bir matris oluşturun ve içinde beyaz bir artı (+) işareti çizin. Arka plan siyah (0), artı işareti beyaz (255) olsun.

---

## Bölüm 2: Renk Uzayları ve RGB

### 2.1 RGB Renk Modeli

Ekrandaki her renkli piksel, üç temel rengin karışımından oluşur:
- **R (Red - Kırmızı)**: 0-255 arası değer
- **G (Green - Yeşil)**: 0-255 arası değer  
- **B (Blue - Mavi)**: 0-255 arası değer

```python
# RGB renk karışımını anlamak
def rgb_karisimi_goster():
    # Boş bir görüntü oluştur
    goruntu = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Farklı bölgelere farklı renkler
    goruntu[0:100, 0:100] = [255, 0, 0]      # Saf Kırmızı
    goruntu[0:100, 100:200] = [0, 255, 0]    # Saf Yeşil
    goruntu[0:100, 200:300] = [0, 0, 255]    # Saf Mavi
    goruntu[0:100, 300:400] = [255, 255, 255] # Beyaz
    
    goruntu[100:200, 0:100] = [255, 255, 0]  # Sarı (R+G)
    goruntu[100:200, 100:200] = [255, 0, 255] # Magenta (R+B)
    goruntu[100:200, 200:300] = [0, 255, 255] # Cyan (G+B)
    goruntu[100:200, 300:400] = [128, 128, 128] # Gri
    
    # OpenCV BGR formatında çalışır, RGB'ye çevir
    goruntu_rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(goruntu_rgb)
    plt.title('RGB Renk Karışımları')
    plt.axis('off')
    plt.show()
    
    return goruntu

# Fonksiyonu çalıştır
renk_ornegi = rgb_karisimi_goster()
```

### 📝 Alıştırma 2: RGB Kanal Ayrıştırma
Verilen bir renkli görüntüyü R, G, B kanallarına ayıran ve her kanalı ayrı ayrı gösteren bir fonksiyon yazın.

```python
def rgb_kanallari_ayir(goruntu_yolu):
    """
    Görüntüyü RGB kanallarına ayırır ve görselleştirir
    """
    # Kodunuzu buraya yazın
    pass
```

---

## Bölüm 3: HSV Renk Uzayı

### 3.1 HSV Nedir ve Neden Önemlidir?

HSV (Hue, Saturation, Value) renk uzayı, renkleri insan algısına daha yakın şekilde temsil eder:

- **Hue (Ton)**: 0-179° arası renk tonu (OpenCV'de)
- **Saturation (Doygunluk)**: 0-255 arası rengin canlılığı
- **Value (Parlaklık)**: 0-255 arası rengin parlaklığı

```python
def hsv_aciklama():
    """HSV değerlerinin etkisini gösterir"""
    
    # Sabit bir renk seçelim (saf kırmızı)
    h_base = 0  # Kırmızı ton
    
    # Saturation etkisi
    sat_ornegi = np.zeros((100, 256, 3), dtype=np.uint8)
    for s in range(256):
        sat_ornegi[:, s] = [h_base, s, 255]
    
    # Value etkisi
    val_ornegi = np.zeros((100, 256, 3), dtype=np.uint8)
    for v in range(256):
        val_ornegi[:, v] = [h_base, 255, v]
    
    # Hue spektrumu
    hue_ornegi = np.zeros((100, 180, 3), dtype=np.uint8)
    for h in range(180):
        hue_ornegi[:, h] = [h, 255, 255]
    
    # HSV'den RGB'ye çevir
    sat_rgb = cv2.cvtColor(sat_ornegi, cv2.COLOR_HSV2RGB)
    val_rgb = cv2.cvtColor(val_ornegi, cv2.COLOR_HSV2RGB)
    hue_rgb = cv2.cvtColor(hue_ornegi, cv2.COLOR_HSV2RGB)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    
    axes[0].imshow(hue_rgb)
    axes[0].set_title('Hue (Ton) Spektrumu - 0-179°')
    axes[0].axis('off')
    
    axes[1].imshow(sat_rgb)
    axes[1].set_title('Saturation (Doygunluk) - Soldan sağa: 0-255')
    axes[1].axis('off')
    
    axes[2].imshow(val_rgb)
    axes[2].set_title('Value (Parlaklık) - Soldan sağa: 0-255')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

hsv_aciklama()
```

### 📝 Alıştırma 3: Renk Tespiti
HSV kullanarak belirli bir rengi tespit eden fonksiyon yazın:

```python
def renk_tespit(goruntu, alt_sinir_hsv, ust_sinir_hsv):
    """
    HSV aralığına göre renk tespiti yapar
    Örnek: Sarı renk için alt_sinir = [20, 100, 100], ust_sinir = [30, 255, 255]
    """
    # Kodunuzu buraya yazın
    pass
```

---

## Bölüm 4: Görüntü Okuma ve Temel İşlemler

### 4.1 Lena Görüntüsü ile Çalışma

Görüntü işlemede klasik test görüntüsü olan Lena'yı kullanalım:

```python
# Lena görüntüsünü yükle (veya başka bir test görüntüsü)
def goruntu_yukle_ve_goster():
    # Örnek görüntü URL'si (Lena)
    # Not: Gerçek uygulamada lokal dosya kullanın
    
    goruntu = cv2.imread('lena.jpg')  # BGR formatında
    goruntu_rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    
    print(f"Görüntü boyutu: {goruntu.shape}")
    print(f"Veri tipi: {goruntu.dtype}")
    print(f"Min değer: {goruntu.min()}, Max değer: {goruntu.max()}")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(goruntu_rgb)
    plt.title('Orijinal Görüntü')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.hist(goruntu.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    
    plt.tight_layout()
    plt.show()
    
    return goruntu

# Test görüntüsü oluşturalım (Lena yerine)
def test_goruntu_olustur():
    """Basit bir test görüntüsü oluşturur"""
    goruntu = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Yüz için ten rengi arka plan
    goruntu[:, :] = [255, 220, 177]
    
    # Sarı şapka
    cv2.ellipse(goruntu, (256, 150), (180, 80), 0, 180, 360, (30, 200, 255), -1)
    
    # Yüz
    cv2.circle(goruntu, (256, 280), 100, (255, 200, 150), -1)
    
    # Gözler
    cv2.circle(goruntu, (220, 260), 15, (50, 50, 50), -1)
    cv2.circle(goruntu, (290, 260), 15, (50, 50, 50), -1)
    
    # Gülümseme
    cv2.ellipse(goruntu, (256, 300), (50, 30), 0, 0, 180, (100, 50, 50), 3)
    
    return goruntu
```

---

## Bölüm 5: Maskeleme İşlemleri

### 5.1 Binary Maske Oluşturma

Maskeleme, görüntünün belirli bölgelerini seçmek veya gizlemek için kullanılır:

```python
def maskeleme_temelleri():
    """Maskeleme işlemlerini sıfırdan kodlayın"""
    
    # Test görüntüsü
    goruntu = test_goruntu_olustur()
    
    # Manuel binary maske oluşturma
    def olustur_binary_maske(goruntu, esik_degeri):
        """
        Gri tonlamalı görüntüden binary maske oluşturur
        """
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        yukseklik, genislik = gri.shape
        maske = np.zeros((yukseklik, genislik), dtype=np.uint8)
        
        for i in range(yukseklik):
            for j in range(genislik):
                if gri[i, j] > esik_degeri:
                    maske[i, j] = 255
                else:
                    maske[i, j] = 0
        
        return maske
    
    # Daha verimli: NumPy kullanarak
    def olustur_binary_maske_hizli(goruntu, esik_degeri):
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        maske = np.where(gri > esik_degeri, 255, 0).astype(np.uint8)
        return maske
    
    maske = olustur_binary_maske_hizli(goruntu, 127)
    
    # Maskeyi uygula
    maskelenmis = cv2.bitwise_and(goruntu, goruntu, mask=maske)
    
    # Sonuçları göster
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Orijinal')
    axes[0].axis('off')
    
    axes[1].imshow(maske, cmap='gray')
    axes[1].set_title('Binary Maske')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(maskelenmis, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Maskelenmiş Görüntü')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return maske, maskelenmis

maske, maskelenmis = maskeleme_temelleri()
```

### 5.2 HSV Tabanlı Maskeleme

```python
def hsv_maskeleme():
    """Belirli renkleri HSV ile maskele"""
    
    goruntu = test_goruntu_olustur()
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    # Sarı renk aralığı (şapka için)
    alt_sari = np.array([20, 100, 100])
    ust_sari = np.array([30, 255, 255])
    
    # Maske oluştur
    maske = cv2.inRange(hsv, alt_sari, ust_sari)
    
    # Sonucu göster
    sonuc = cv2.bitwise_and(goruntu, goruntu, mask=maske)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
    plt.title('Sarı Maskeleme')
    plt.axis('off')
    
    plt.show()
    
    return maske

sari_maske = hsv_maskeleme()
```

### 📝 Alıştırma 4: Çoklu Renk Maskeleme
Birden fazla renk aralığını tespit eden ve maskeleyen fonksiyon yazın:

```python
def coklu_renk_maskele(goruntu, renk_araliklari):
    """
    renk_araliklari: [(alt_hsv1, ust_hsv1), (alt_hsv2, ust_hsv2), ...]
    """
    # Kodunuzu buraya yazın
    pass
```

---

## Bölüm 6: Kenar Algılama Algoritmaları

### 6.1 Gradyan Tabanlı Kenar Algılama (Sıfırdan)

```python
def sobel_filtresi_manuel():
    """Sobel filtresini sıfırdan kodlayın"""
    
    # Sobel çekirdekleri
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    
    def konvolusyon_2d(goruntu, cekirdek):
        """2D konvolüsyon işlemi"""
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        
        yukseklik, genislik = goruntu.shape
        k_yukseklik, k_genislik = cekirdek.shape
        
        # Padding ekle
        pad_y = k_yukseklik // 2
        pad_x = k_genislik // 2
        
        padli_goruntu = np.pad(goruntu, ((pad_y, pad_y), (pad_x, pad_x)), 
                                mode='constant', constant_values=0)
        
        # Çıktı matrisi
        cikti = np.zeros((yukseklik, genislik))
        
        # Konvolüsyon
        for i in range(yukseklik):
            for j in range(genislik):
                bolge = padli_goruntu[i:i+k_yukseklik, j:j+k_genislik]
                cikti[i, j] = np.sum(bolge * cekirdek)
        
        return cikti
    
    # Test görüntüsü
    goruntu = test_goruntu_olustur()
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    
    # Gradyanları hesapla
    gx = konvolusyon_2d(gri, sobel_x)
    gy = konvolusyon_2d(gri, sobel_y)
    
    # Gradyan büyüklüğü
    gradyan_buyukluk = np.sqrt(gx**2 + gy**2)
    gradyan_buyukluk = np.clip(gradyan_buyukluk, 0, 255).astype(np.uint8)
    
    # Gradyan yönü
    gradyan_yon = np.arctan2(gy, gx)
    
    # Sonuçları göster
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gri, cmap='gray')
    axes[0, 0].set_title('Orijinal (Gri)')
    
    axes[0, 1].imshow(gx, cmap='gray')
    axes[0, 1].set_title('Sobel X')
    
    axes[0, 2].imshow(gy, cmap='gray')
    axes[0, 2].set_title('Sobel Y')
    
    axes[1, 0].imshow(gradyan_buyukluk, cmap='gray')
    axes[1, 0].set_title('Gradyan Büyüklüğü')
    
    axes[1, 1].imshow(gradyan_yon, cmap='hsv')
    axes[1, 1].set_title('Gradyan Yönü')
    
    # Eşikleme ile kenar
    esik = 50
    kenarlar = np.where(gradyan_buyukluk > esik, 255, 0).astype(np.uint8)
    axes[1, 2].imshow(kenarlar, cmap='gray')
    axes[1, 2].set_title(f'Kenarlar (Eşik={esik})')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gradyan_buyukluk

sobel_sonuc = sobel_filtresi_manuel()
```

### 6.2 Canny Kenar Algılama (Basitleştirilmiş)

```python
def canny_algoritmasi_aciklama():
    """Canny algoritmasının adımları"""
    
    print("CANNY KENAR ALGILAMA ADIMLARI:")
    print("="*50)
    print("1. Gaussian Blur ile gürültü azaltma")
    print("2. Sobel filtresi ile gradyan hesaplama")
    print("3. Non-maximum suppression (İnce kenarlar)")
    print("4. Double thresholding (Çift eşikleme)")
    print("5. Edge tracking by hysteresis")
    print("="*50)
    
    def basit_canny(goruntu, alt_esik, ust_esik):
        """Basitleştirilmiş Canny implementasyonu"""
        
        if len(goruntu.shape) == 3:
            gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        else:
            gri = goruntu.copy()
        
        # 1. Gaussian blur
        blurlu = cv2.GaussianBlur(gri, (5, 5), 1.4)
        
        # 2. Gradyan hesaplama (Sobel)
        gx = cv2.Sobel(blurlu, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurlu, cv2.CV_64F, 0, 1, ksize=3)
        
        buyukluk = np.sqrt(gx**2 + gy**2)
        yon = np.arctan2(gy, gx)
        
        # 3. Non-maximum suppression (basitleştirilmiş)
        yukseklik, genislik = buyukluk.shape
        suppressed = np.zeros_like(buyukluk)
        
        yon_derece = np.rad2deg(yon) % 180
        
        for i in range(1, yukseklik-1):
            for j in range(1, genislik-1):
                # Yön kuantizasyonu (0, 45, 90, 135 derece)
                if (0 <= yon_derece[i,j] < 22.5) or (157.5 <= yon_derece[i,j] <= 180):
                    # Yatay kenar
                    if (buyukluk[i,j] >= buyukluk[i,j-1]) and (buyukluk[i,j] >= buyukluk[i,j+1]):
                        suppressed[i,j] = buyukluk[i,j]
                elif 22.5 <= yon_derece[i,j] < 67.5:
                    # Diagonal kenar
                    if (buyukluk[i,j] >= buyukluk[i-1,j+1]) and (buyukluk[i,j] >= buyukluk[i+1,j-1]):
                        suppressed[i,j] = buyukluk[i,j]
                elif 67.5 <= yon_derece[i,j] < 112.5:
                    # Dikey kenar
                    if (buyukluk[i,j] >= buyukluk[i-1,j]) and (buyukluk[i,j] >= buyukluk[i+1,j]):
                        suppressed[i,j] = buyukluk[i,j]
                else:
                    # Diagonal kenar
                    if (buyukluk[i,j] >= buyukluk[i-1,j-1]) and (buyukluk[i,j] >= buyukluk[i+1,j+1]):
                        suppressed[i,j] = buyukluk[i,j]
        
        # 4. Double thresholding
        guclu_kenar = 255
        zayif_kenar = 50
        
        kenarlar = np.zeros_like(suppressed, dtype=np.uint8)
        kenarlar[suppressed > ust_esik] = guclu_kenar
        kenarlar[(suppressed > alt_esik) & (suppressed <= ust_esik)] = zayif_kenar
        
        return kenarlar
    
    # Test
    goruntu = test_goruntu_olustur()
    manuel_canny = basit_canny(goruntu, 50, 150)
    opencv_canny = cv2.Canny(cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY), 50, 150)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Orijinal')
    
    axes[1].imshow(manuel_canny, cmap='gray')
    axes[1].set_title('Manuel Canny')
    
    axes[2].imshow(opencv_canny, cmap='gray')
    axes[2].set_title('OpenCV Canny')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return manuel_canny, opencv_canny

canny_algoritmasi_aciklama()
```

### 📝 Alıştırma 5: Adaptif Kenar Algılama
Görüntünün farklı bölgelerinde farklı eşik değerleri kullanan adaptif kenar algılama fonksiyonu yazın:

```python
def adaptif_kenar_algilama(goruntu, pencere_boyutu=50):
    """
    Görüntüyü pencere_boyutu x pencere_boyutu bloklara böl
    Her blok için optimal eşik değeri hesapla
    """
    # Kodunuzu buraya yazın
    pass
```

---

## Bölüm 7: Morfolojik İşlemler

### 7.1 Erozyon ve Dilatasyon

```python
def morfolojik_islemler():
    """Morfolojik işlemleri sıfırdan kodlayın"""
    
    def erozyon(goruntu, cekirdek):
        """Erozyon işlemi"""
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
            _, goruntu = cv2.threshold(goruntu, 127, 255, cv2.THRESH_BINARY)
        
        yukseklik, genislik = goruntu.shape
        k_yukseklik, k_genislik = cekirdek.shape
        
        pad_y = k_yukseklik // 2
        pad_x = k_genislik // 2
        
        padli = np.pad(goruntu, ((pad_y, pad_y), (pad_x, pad_x)), 
                       mode='constant', constant_values=0)
        
        sonuc = np.zeros_like(goruntu)
        
        for i in range(yukseklik):
            for j in range(genislik):
                bolge = padli[i:i+k_yukseklik, j:j+k_genislik]
                # Erozyon: Tüm pikseller beyazsa merkez beyaz
                if np.all(bolge[cekirdek == 1] == 255):
                    sonuc[i, j] = 255
        
        return sonuc
    
    def dilatasyon(goruntu, cekirdek):
        """Dilatasyon işlemi"""
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
            _, goruntu = cv2.threshold(goruntu, 127, 255, cv2.THRESH_BINARY)
        
        yukseklik, genislik = goruntu.shape
        k_yukseklik, k_genislik = cekirdek.shape
        
        pad_y = k_yukseklik // 2
        pad_x = k_genislik // 2
        
        padli = np.pad(goruntu, ((pad_y, pad_y), (pad_x, pad_x)), 
                       mode='constant', constant_values=0)
        
        sonuc = np.zeros_like(goruntu)
        
        for i in range(yukseklik):
            for j in range(genislik):
                bolge = padli[i:i+k_yukseklik, j:j+k_genislik]
                # Dilatasyon: En az bir piksel beyazsa merkez beyaz
                if np.any(bolge[cekirdek == 1] == 255):
                    sonuc[i, j] = 255
        
        return sonuc
    
    # Test görüntüsü oluştur
    test = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(test, (30, 30), (70, 70), 255, -1)
    cv2.circle(test, (50, 50), 10, 0, -1)  # İçinde boşluk
    
    # Çekirdek
    cekirdek = np.ones((3, 3), dtype=np.uint8)
    
    # İşlemleri uygula
    erozyonlu = erozyon(test, cekirdek)
    dilatasyonlu = dilatasyon(test, cekirdek)
    
    # Opening (Erozyon + Dilatasyon)
    opening = dilatasyon(erozyon(test, cekirdek), cekirdek)
    
    # Closing (Dilatasyon + Erozyon)
    closing = erozyon(dilatasyon(test, cekirdek), cekirdek)
    
    # Sonuçları göster
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    basliklar = ['Orijinal', 'Erozyon', 'Dilatasyon', 
                  'Opening', 'Closing', 'Gradient']
    
    gradient = dilatasyonlu - erozyonlu
    
    gorseller = [test, erozyonlu, dilatasyonlu, opening, closing, gradient]
    
    for ax, baslik, gorsel in zip(axes.flat, basliklar, gorseller):
        ax.imshow(gorsel, cmap='gray')
        ax.set_title(baslik)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return test, erozyonlu, dilatasyonlu

test_goruntu, erode, dilate = morfolojik_islemler()
```

---

## Bölüm 8: Histogram İşlemleri

### 8.1 Histogram Eşitleme

```python
def histogram_esitleme_manuel():
    """Histogram eşitlemeyi sıfırdan kodlayın"""
    
    def histogram_esitle(goruntu):
        """
        Histogram eşitleme algoritması
        """
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        
        # 1. Histogram hesapla
        histogram = np.zeros(256)
        yukseklik, genislik = goruntu.shape
        toplam_piksel = yukseklik * genislik
        
        for i in range(yukseklik):
            for j in range(genislik):
                histogram[goruntu[i, j]] += 1
        
        # 2. Normalize et (PMF - Probability Mass Function)
        pmf = histogram / toplam_piksel
        
        # 3. CDF (Cumulative Distribution Function) hesapla
        cdf = np.zeros(256)
        cdf[0] = pmf[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + pmf[i]
        
        # 4. Transfer fonksiyonu oluştur
        transfer = np.round(cdf * 255).astype(np.uint8)
        
        # 5. Yeni görüntüyü oluştur
        esitlenmis = np.zeros_like(goruntu)
        for i in range(yukseklik):
            for j in range(genislik):
                esitlenmis[i, j] = transfer[goruntu[i, j]]
        
        return esitlenmis, histogram, cdf
    
    # Test
    goruntu = test_goruntu_olustur()
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    
    # Düşük kontrastlı görüntü oluştur
    dusuk_kontrast = (gri * 0.3 + 100).astype(np.uint8)
    
    # Histogram eşitle
    esitlenmis, hist_orijinal, cdf = histogram_esitle(dusuk_kontrast)
    
    # OpenCV ile karşılaştır
    opencv_esitlenmis = cv2.equalizeHist(dusuk_kontrast)
    
    # Sonuçları göster
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Orijinal
    axes[0, 0].imshow(dusuk_kontrast, cmap='gray')
    axes[0, 0].set_title('Düşük Kontrast')
    
    axes[1, 0].hist(dusuk_kontrast.ravel(), bins=256, range=[0, 256])
    axes[1, 0].set_title('Histogram')
    
    # Manuel eşitleme
    axes[0, 1].imshow(esitlenmis, cmap='gray')
    axes[0, 1].set_title('Manuel Eşitleme')
    
    axes[1, 1].hist(esitlenmis.ravel(), bins=256, range=[0, 256])
    axes[1, 1].set_title('Eşitlenmiş Histogram')
    
    # OpenCV eşitleme
    axes[0, 2].imshow(opencv_esitlenmis, cmap='gray')
    axes[0, 2].set_title('OpenCV Eşitleme')
    
    axes[1, 2].hist(opencv_esitlenmis.ravel(), bins=256, range=[0, 256])
    axes[1, 2].set_title('OpenCV Histogram')
    
    # CDF
    axes[0, 3].plot(cdf)
    axes[0, 3].set_title('CDF')
    axes[0, 3].set_xlabel('Piksel Değeri')
    axes[0, 3].set_ylabel('Kümülatif Olasılık')
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return esitlenmis

esitlenmis_goruntu = histogram_esitleme_manuel()
```

### 📝 Alıştırma 6: CLAHE (Contrast Limited Adaptive Histogram Equalization)
Görüntüyü bloklara bölüp her blokta ayrı histogram eşitleme yapan CLAHE algoritmasını basit haliyle kodlayın:

```python
def basit_clahe(goruntu, blok_boyutu=8):
    """
    Adaptif histogram eşitleme
    """
    # Kodunuzu buraya yazın
    pass
```

---

## Bölüm 9: Görüntü Filtreleme

### 9.1 Blur (Bulanıklaştırma) Filtreleri

```python
def blur_filtreleri():
    """Farklı blur tekniklerini sıfırdan kodlayın"""
    
    def box_filter(goruntu, cekirdek_boyutu):
        """Basit ortalama filtresi"""
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        
        k = cekirdek_boyutu
        cekirdek = np.ones((k, k)) / (k * k)
        
        yukseklik, genislik = goruntu.shape
        pad = k // 2
        
        padli = np.pad(goruntu, pad, mode='edge')
        sonuc = np.zeros_like(goruntu, dtype=np.float32)
        
        for i in range(yukseklik):
            for j in range(genislik):
                bolge = padli[i:i+k, j:j+k]
                sonuc[i, j] = np.sum(bolge * cekirdek)
        
        return sonuc.astype(np.uint8)
    
    def gaussian_kernel(boyut, sigma):
        """Gaussian çekirdeği oluştur"""
        cekirdek = np.zeros((boyut, boyut))
        merkez = boyut // 2
        
        toplam = 0
        for i in range(boyut):
            for j in range(boyut):
                x = i - merkez
                y = j - merkez
                cekirdek[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                toplam += cekirdek[i, j]
        
        return cekirdek / toplam
    
    def gaussian_blur(goruntu, cekirdek_boyutu, sigma):
        """Gaussian blur uygula"""
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        
        cekirdek = gaussian_kernel(cekirdek_boyutu, sigma)
        
        yukseklik, genislik = goruntu.shape
        k = cekirdek_boyutu
        pad = k // 2
        
        padli = np.pad(goruntu, pad, mode='edge')
        sonuc = np.zeros_like(goruntu, dtype=np.float32)
        
        for i in range(yukseklik):
            for j in range(genislik):
                bolge = padli[i:i+k, j:j+k]
                sonuc[i, j] = np.sum(bolge * cekirdek)
        
        return sonuc.astype(np.uint8)
    
    def median_filter(goruntu, pencere_boyutu):
        """Medyan filtresi - gürültü giderme için"""
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        
        yukseklik, genislik = goruntu.shape
        pad = pencere_boyutu // 2
        
        padli = np.pad(goruntu, pad, mode='edge')
        sonuc = np.zeros_like(goruntu)
        
        for i in range(yukseklik):
            for j in range(genislik):
                bolge = padli[i:i+pencere_boyutu, j:j+pencere_boyutu]
                sonuc[i, j] = np.median(bolge)
        
        return sonuc.astype(np.uint8)
    
    # Test
    goruntu = test_goruntu_olustur()
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    
    # Gürültü ekle
    gurultulu = gri.copy()
    gurultu = np.random.randint(0, 50, gri.shape)
    gurultulu = np.clip(gurultulu.astype(int) + gurultu, 0, 255).astype(np.uint8)
    
    # Filtreleri uygula
    box_blurlu = box_filter(gurultulu, 5)
    gaussian_blurlu = gaussian_blur(gurultulu, 5, 1.0)
    median_blurlu = median_filter(gurultulu, 5)
    
    # Sonuçları göster
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gri, cmap='gray')
    axes[0, 0].set_title('Orijinal')
    
    axes[0, 1].imshow(gurultulu, cmap='gray')
    axes[0, 1].set_title('Gürültülü')
    
    axes[0, 2].imshow(box_blurlu, cmap='gray')
    axes[0, 2].set_title('Box Filter')
    
    axes[1, 0].imshow(gaussian_blurlu, cmap='gray')
    axes[1, 0].set_title('Gaussian Blur')
    
    axes[1, 1].imshow(median_blurlu, cmap='gray')
    axes[1, 1].set_title('Median Filter')
    
    # Gaussian çekirdeği görselleştir
    gaussian_cekirdek = gaussian_kernel(21, 3)
    axes[1, 2].imshow(gaussian_cekirdek, cmap='hot')
    axes[1, 2].set_title('Gaussian Kernel (σ=3)')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return box_blurlu, gaussian_blurlu, median_blurlu

blur_filtreleri()
```

---

## Bölüm 10: İleri Seviye Konular

### 10.1 Görüntü Piramitleri

```python
def goruntu_piramidi():
    """Gaussian ve Laplacian piramitleri"""
    
    def gaussian_pyramid(goruntu, seviye=5):
        """Gaussian piramidi oluştur"""
        piramit = [goruntu]
        
        for i in range(seviye - 1):
            # Blur ve downsample
            blurlu = cv2.GaussianBlur(piramit[-1], (5, 5), 1)
            kucuk = blurlu[::2, ::2]  # 2x downsampling
            piramit.append(kucuk)
        
        return piramit
    
    def laplacian_pyramid(gaussian_piramit):
        """Laplacian piramidi oluştur"""
        laplacian_piramit = []
        
        for i in range(len(gaussian_piramit) - 1):
            # Upsampling
            buyutulmus = cv2.resize(gaussian_piramit[i+1], 
                                    (gaussian_piramit[i].shape[1], 
                                     gaussian_piramit[i].shape[0]))
            
            # Laplacian = orijinal - büyütülmüş
            laplacian = cv2.subtract(gaussian_piramit[i], buyutulmus)
            laplacian_piramit.append(laplacian)
        
        # Son seviye Gaussian'ın kendisi
        laplacian_piramit.append(gaussian_piramit[-1])
        
        return laplacian_piramit
    
    # Test
    goruntu = test_goruntu_olustur()
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    
    gauss_piramit = gaussian_pyramid(gri, 5)
    lap_piramit = laplacian_pyramid(gauss_piramit)
    
    # Görselleştir
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, img in enumerate(gauss_piramit):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Gaussian L{i}')
        axes[0, i].axis('off')
    
    for i, img in enumerate(lap_piramit):
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Laplacian L{i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gauss_piramit, lap_piramit

gauss_p, lap_p = goruntu_piramidi()
```

### 10.2 Template Matching (Şablon Eşleştirme)

```python
def template_matching():
    """Görüntüde şablon arama"""
    
    def normalized_cross_correlation(goruntu, sablon):
        """NCC ile şablon eşleştirme"""
        if len(goruntu.shape) == 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        if len(sablon.shape) == 3:
            sablon = cv2.cvtColor(sablon, cv2.COLOR_BGR2GRAY)
        
        h, w = sablon.shape
        H, W = goruntu.shape
        
        korelasyon_haritasi = np.zeros((H - h + 1, W - w + 1))
        
        # Şablon normalizasyonu
        sablon_ort = np.mean(sablon)
        sablon_norm = sablon - sablon_ort
        sablon_std = np.sqrt(np.sum(sablon_norm**2))
        
        for i in range(H - h + 1):
            for j in range(W - w + 1):
                pencere = goruntu[i:i+h, j:j+w]
                pencere_ort = np.mean(pencere)
                pencere_norm = pencere - pencere_ort
                
                pay = np.sum(pencere_norm * sablon_norm)
                payda = np.sqrt(np.sum(pencere_norm**2)) * sablon_std
                
                if payda != 0:
                    korelasyon_haritasi[i, j] = pay / payda
        
        return korelasyon_haritasi
    
    # Test
    buyuk_goruntu = np.zeros((300, 400), dtype=np.uint8)
    buyuk_goruntu[50:100, 50:100] = 255  # Beyaz kare
    buyuk_goruntu[150:200, 250:300] = 255  # Başka bir beyaz kare
    buyuk_goruntu[200:250, 100:150] = 128  # Gri kare
    
    sablon = np.ones((50, 50), dtype=np.uint8) * 255  # Beyaz kare şablonu
    
    # Template matching
    korelasyon = normalized_cross_correlation(buyuk_goruntu, sablon)
    
    # En iyi eşleşmeyi bul
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(korelasyon)
    
    # Sonuçları göster
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(buyuk_goruntu, cmap='gray')
    axes[0].set_title('Ana Görüntü')
    axes[0].axis('off')
    
    axes[1].imshow(sablon, cmap='gray')
    axes[1].set_title('Şablon')
    axes[1].axis('off')
    
    axes[2].imshow(korelasyon, cmap='hot')
    axes[2].set_title('Korelasyon Haritası')
    axes[2].plot(max_loc[0], max_loc[1], 'b*', markersize=15)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"En iyi eşleşme konumu: {max_loc}")
    print(f"Korelasyon değeri: {max_val:.3f}")
    
    return korelasyon

template_matching()
```

### 📝 Alıştırma 7: Hough Transform
Görüntüde çizgi tespiti için basit Hough Transform implementasyonu:

```python
def hough_lines(kenar_goruntusu, rho_resolution=1, theta_resolution=np.pi/180):
    """
    Hough transform ile çizgi tespiti
    kenar_goruntusu: Binary kenar görüntüsü
    """
    # Kodunuzu buraya yazın
    # İpucu: Accumulator matrix oluşturun
    # Her kenar pikseli için tüm olası (rho, theta) değerlerini hesaplayın
    pass
```

---

## Bölüm 11: Performans Optimizasyonu

### 11.1 Vektörleştirme ve NumPy Kullanımı

```python
def performans_karsilastirma():
    """Loop vs NumPy performans karşılaştırması"""
    import time
    
    goruntu = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    
    # Yavaş yöntem (loop)
    def brightness_loop(img, value):
        result = np.zeros_like(img)
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                new_val = img[i, j] + value
                result[i, j] = min(max(new_val, 0), 255)
        return result
    
    # Hızlı yöntem (NumPy)
    def brightness_numpy(img, value):
        return np.clip(img.astype(int) + value, 0, 255).astype(np.uint8)
    
    # Zamanlama
    start = time.time()
    result_loop = brightness_loop(goruntu, 50)
    time_loop = time.time() - start
    
    start = time.time()
    result_numpy = brightness_numpy(goruntu, 50)
    time_numpy = time.time() - start
    
    print(f"Loop süresi: {time_loop:.3f} saniye")
    print(f"NumPy süresi: {time_numpy:.3f} saniye")
    print(f"Hızlanma: {time_loop/time_numpy:.1f}x")
    
    # Sonuçların aynı olduğunu kontrol et
    print(f"Sonuçlar aynı mı? {np.array_equal(result_loop, result_numpy)}")

performans_karsilastirma()
```

---

## PROJE ÖDEVLERİ

### Proje 1: Otomatik Renk Düzeltme
Bir görüntünün renk dengesini otomatik düzelten sistem yazın:
- Beyaz dengesi ayarlama
- Histogram eşitleme
- Kontrast ve parlaklık optimizasyonu

### Proje 2: Belge Tarayıcı
Kamera ile çekilen belge fotoğrafını düzelten sistem:
- Kenar tespiti ile belge sınırlarını bulma
- Perspektif düzeltme
- Arka plan temizleme
- Metin netleştirme

### Proje 3: Yüz Bulanıklaştırma
Görüntüdeki yüzleri tespit edip bulanıklaştıran sistem:
- Ten rengi tespiti (HSV)
- Yüz bölgesi segmentasyonu
- Seçici bulanıklaştırma

### Proje 4: Görüntü Mozaikleme
Birden fazla görüntüyü birleştirerek panorama oluşturma:
- Özellik noktaları bulma (Harris corner detector)
- Görüntü eşleştirme
- Homografi hesaplama
- Görüntü birleştirme

### Proje 5: OCR Ön İşleme
Metin tanıma için görüntü ön işleme:
- Gürültü giderme
- Binarizasyon (Otsu thresholding)
- Deskewing (eğrilik düzeltme)
- Karakter segmentasyonu

---

## Kaynaklar ve İleri Okuma

1. **Temel Kaynaklar:**
   - Digital Image Processing (Gonzalez & Woods)
   - Computer Vision: Algorithms and Applications (Szeliski)
   - OpenCV-Python Tutorials

2. **Pratik Kaynaklar:**
   - PyImageSearch Blog
   - OpenCV Documentation
   - scikit-image Documentation

3. **İleri Seviye Konular:**
   - Deep Learning for Computer Vision
   - Feature Detection and Matching
   - 3D Computer Vision
   - Video Processing

---

## Notlar

- Her bölümde verilen kodları kendi bilgisayarınızda çalıştırın
- Alıştırmaları tamamlayın ve farklı parametrelerle deneyin
- Kendi görüntüleriniz üzerinde testler yapın
- Performans optimizasyonuna dikkat edin
- Hata ayıklama için ara sonuçları görselleştirin

**Başarılar! 🚀**
