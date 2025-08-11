import cv2
import numpy as np


def basit_clahe(goruntu, blok_boyutu=8, clip_limit=2.0):
    """
    Basit CLAHE implementasyonu

    Parametreler:
        goruntu: Gri tonlamalı veya BGR görüntü
        blok_boyutu: Bölünecek blok sayısı (varsayılan: 8x8 blok)
        clip_limit: Kontrast kırpma limiti (0-3 arası)

    Döndürür:
        Kontrastı iyileştirilmiş görüntü
    """
    # 1. Görüntüyü gri tonlamalıya çevir
    if len(goruntu.shape) == 3:
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    else:
        gri = goruntu.copy()

    # 2. Görüntü boyutlarını ve blok özelliklerini hesapla
    yukseklik, genislik = gri.shape
    blok_h = yukseklik // blok_boyutu
    blok_w = genislik // blok_boyutu

    # 3. Çıktı görüntüsünü oluştur
    cikti = np.zeros_like(gri)

    # 4. Her blok için işlem yap
    for i in range(blok_boyutu):
        for j in range(blok_boyutu):
            # Blok sınırlarını belirle
            y_start = i * blok_h
            y_end = (i + 1) * blok_h if i != blok_boyutu - 1 else yukseklik
            x_start = j * blok_w
            x_end = (j + 1) * blok_w if j != blok_boyutu - 1 else genislik

            blok = gri[y_start:y_end, x_start:x_end]

            # 5. Histogram eşitleme (CLAHE mantığı)
            # 5a. Histogram hesapla
            hist, _ = np.histogram(blok.flatten(), bins=256, range=[0, 256])

            # 5b. Histogramı kırp (clip limit)
            clip_threshold = clip_limit * hist.mean()
            excess = np.sum(np.maximum(hist - clip_threshold, 0))
            hist = np.minimum(hist, clip_threshold)
            hist += excess // 256  # Kırpılan değerleri eşit dağıt

            # 5c. CDF hesapla
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            cdf = cdf.astype(np.uint8)

            # 5d. Blokta eşitleme uygula
            blok_esitlenmis = cdf[blok]
            cikti[y_start:y_end, x_start:x_end] = blok_esitlenmis

    return cikti


# Görüntüyü yükle
goruntu = cv2.imread('lena.jpg')

gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)  #

# CLAHE uygula
clahe_sonuc = basit_clahe(goruntu, blok_boyutu=8, clip_limit=2.0)

# OpenCV'nin CLAHE'si ile karşılaştır
cv_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gri)

# Sonuçları göster
cv2.imshow('Orijinal', goruntu)
cv2.imshow('Basit CLAHE', clahe_sonuc)
cv2.imshow('OpenCV CLAHE', cv_clahe)
cv2.waitKey(0)