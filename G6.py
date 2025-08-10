import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from G4 import test_goruntu_olustur


def sobel_filtresi_manuel():
    """Sobel filtresini sıfırdan kodlayın"""

    # Sobel çekirdekleri
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

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
                bolge = padli_goruntu[i:i + k_yukseklik, j:j + k_genislik]
                cikti[i, j] = np.sum(bolge * cekirdek)

        return cikti

    # Test görüntüsü
    goruntu = test_goruntu_olustur()
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    # Gradyanları hesapla
    gx = konvolusyon_2d(gri, sobel_x)
    gy = konvolusyon_2d(gri, sobel_y)

    # Gradyan büyüklüğü
    gradyan_buyukluk = np.sqrt(gx ** 2 + gy ** 2)
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