import cv2
import numpy as np


def adaptif_kenar_algilama(goruntu, pencere_boyutu=50):

    # Görüntüyü gri tonlamalı yap
    if len(goruntu.shape) == 3:
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    else:
        gri = goruntu.copy()

    # Görüntü boyutlarını al
    yukseklik, genislik = gri.shape

    # Çıktı görüntüsünü oluştur
    kenar_haritasi = np.zeros_like(gri)

    # Görüntüyü pencere_boyutu x pencere_boyutu bloklara böl
    for y in range(0, yukseklik, pencere_boyutu):
        for x in range(0, genislik, pencere_boyutu):
            # Blok sınırlarını belirle
            y1 = min(y + pencere_boyutu, yukseklik)
            x1 = min(x + pencere_boyutu, genislik)

            # Blok'u al
            blok = gri[y:y1, x:x1]

            # Blok için eşik değerlerini hesapla
            ortalama = np.mean(blok)
            std = np.std(blok)

            # Dinamik eşik değerleri
            alt_esik = max(0, int(ortalama - std))
            ust_esik = min(255, int(ortalama + std))

            # Blok için Canny uygula
            blok_kenarlar = cv2.Canny(blok, alt_esik, ust_esik)

            # Sonuçları ana haritaya ekle
            kenar_haritasi[y:y1, x:x1] = blok_kenarlar

    return kenar_haritasi


# Görüntüyü yükle
goruntu = cv2.imread('lena.jpg')

# Adaptif kenar algılama uygula
kenarlar = adaptif_kenar_algilama(goruntu, pencere_boyutu=50)

# Sonuçları göster
cv2.imshow('Orijinal', goruntu)
cv2.imshow('Adaptif Kenarlar', kenarlar)
cv2.waitKey(0)
cv2.destroyAllWindows()