import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from G4 import test_goruntu_olustur

def canny_algoritmasi_man():
    #1-goruntuyu griye cevir
    #2-GaussianBlur ile gurultuyu azalt
    #3-kenar gucu ve yonunu bul sobel ile Gradyan hesaplama
    #4-Kenarlari incelt(non-maximum supperssion)
    #5-Double thresholding(Cift esikleme)

  #1-
    def basit_canny(goruntu, alt_esik, ust_esik):
        if len(goruntu.shape)== 3:
            gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        else:
            gri = goruntu.copy()

        # 2-
        blur = cv2.GaussianBlur(gri, (5,5), 1.4)

        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

        buyukluk = np.sqrt(gx ** 2 + gy ** 2)
        yon = np.arctan2(gy, gx)

        # 3-
        yukseklik, genislik = buyukluk.shape
        suppressed = np.zeros_like( buyukluk)  # burda verilen matrisle ayni boyutlarda 0 ile dolu yeni bir mayris olusturyoruz
        yon_derece = np.rad2deg(yon) % 180

        for i in range(1, yukseklik - 1):
            for j in range(1, genislik - 1):
                # Yön kuantizasyonu (0, 45, 90, 135 derece)
                if (0 <= yon_derece[i, j] < 22.5) or (157.5 <= yon_derece[i, j] <= 180):
                    # Yatay kenar
                    if (buyukluk[i, j] >= buyukluk[i, j - 1]) and (buyukluk[i, j] >= buyukluk[i, j + 1]):
                        suppressed[i, j] = buyukluk[i, j]
                elif 22.5 <= yon_derece[i, j] < 67.5:
                    # Diagonal kenar
                    if (buyukluk[i, j] >= buyukluk[i - 1, j + 1]) and (buyukluk[i, j] >= buyukluk[i + 1, j - 1]):
                        suppressed[i, j] = buyukluk[i, j]
                elif 67.5 <= yon_derece[i, j] < 112.5:
                    # Dikey kenar
                    if (buyukluk[i, j] >= buyukluk[i - 1, j]) and (buyukluk[i, j] >= buyukluk[i + 1, j]):
                        suppressed[i, j] = buyukluk[i, j]
                else:
                    # Diagonal kenar
                    if (buyukluk[i, j] >= buyukluk[i - 1, j - 1]) and (buyukluk[i, j] >= buyukluk[i + 1, j + 1]):
                        suppressed[i, j] = buyukluk[i, j]

        # 4-
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


canny_algoritmasi_man()




