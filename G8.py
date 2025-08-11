
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from G4 import test_goruntu_olustur


def histogram_esitleme_man():
    def histogram_esitle(goruntu):
        if len(goruntu.shape)== 3:
            goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)


        histogram = np.zeros(256)
        yukseklik,genislik = goruntu.shape
        toplam_piksel = yukseklik*genislik

        for i in range(yukseklik): #0-255 arasi tum tonlar icin sayac
            for j in range(genislik):
                histogram[goruntu[i,j]]+= 1


        pmf = histogram /toplam_piksel #olasılık kütle fonksiyonu.

        cdf = np.zeros(256)
        cdf[0] = pmf[0]
        for i in range(1,256):
            cdf[i]=cdf[i-1]+pmf[i]


        transfer = np.round(cdf*255).astype(np.uint8)

        esitlenmis = np.zeros_like(goruntu)
        for i in range(yukseklik):
            for j in range(genislik):
                esitlenmis[i,j]=transfer[goruntu[i,j]]

        return esitlenmis, histogram,cdf


    goruntu = test_goruntu_olustur()
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    dusuk_kontrast = (gri*0.3+100).astype(np.uint8)

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


esitlenmis_goruntu = histogram_esitleme_man()



