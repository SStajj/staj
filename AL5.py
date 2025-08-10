import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PIL.ImageFilter import GaussianBlur


def adaptif_kenar_algilama(goruntu, pencere_boyutu=50):
    #1-goruntuyu griye ceviririm
    if len(goruntu.shape)==3:
        gri =cv2.cvtColor(goruntu,cv2.COLOR_BGR2GRAY)
    else:
        gri=goruntu.copy()
    #2-
    blurlu = cv2.GaussianBlur(gri, (5, 5), 1.2)

    gx =cv2.Sobel(blurlu, cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(blurlu, cv2.CV_64F,0,1, ksize=3)

    size = np.sqrt(gx**2 + gy**2)
    yonu = np.arctan2(gy,gx)

    #3-
    height,weight=size.shape
    sup =np.zeros_like(size)

    derece = np.rad2deg(yonu)%180

    for i in range(1, height - 1):
        for j in range(1, weight - 1):
            if (0 <= derece[i, j] < 22.5) or (157.5 <= derece[i, j] <= 180):
                if (size[i, j] >= size[i, j - 1]) and (size[i, j] >= size[i, j + 1]):
                    sup[i, j] = size[i, j]
            elif 22.5 <= derece[i, j] < 67.5:
                if (size[i, j] >= size[i - 1, j + 1]) and (size[i, j] >= size[i + 1, j - 1]):
                    sup[i, j] = size[i, j]
            elif 67.5 <= derece[i, j] < 112.5:
                if (size[i, j] >= size[i - 1, j]) and (size[i, j] >= size[i + 1, j]):
                    sup[i, j] = size[i, j]
            else:
                if (size[i, j] >= size[i - 1, j - 1]) and (size[i, j] >= size[i + 1, j + 1]):
                    sup[i, j] = size[i, j]

        # 4 -
    g_kenar = 255
    z_kenar = 50
    kenar = np.zeros_like(sup, dtype=np.uint8)

    for y in range(0, height , pencere_boyutu):
        for x in range(0, weight, pencere_boyutu):
            y1,y2 = y, min(y +pencere_boyutu,height)
            x1,x2 = x, min(x+ pencere_boyutu, weight)

            blok = sup[y1:y2, x1:x2]
            alt_esik = np.percentile(blok, 30)
            ust_esik = np.percentile(blok, 70)

            kenar[y1:y2, x1:x2][blok > ust_esik] = g_kenar
            kenar[y1:y2, x1:x2][(blok > alt_esik) & (blok <= ust_esik)] = z_kenar


    return kenar

goruntu = cv2.imread("C:/Users/sudea/Downloads/lena (1).png")
sonuc = adaptif_kenar_algilama(goruntu, pencere_boyutu=50)

plt.imshow(sonuc, cmap='gray')
plt.title("Adaptif Kenar Algılama")
plt.axis("off")
plt.show()






