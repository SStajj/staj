
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
                bolge = padli[i: i +k_yukseklik, j: j +k_genislik]
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
                bolge = padli[i: i +k_yukseklik, j: j +k_genislik]
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