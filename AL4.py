import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from G4 import test_goruntu_olustur


def coklu_renk_maskeleme(goruntu, renk_araliklari):
    plt.close('all')
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)

    toplam_maske =np.zeros(hsv.shape[:2], dtype = np.uint8)

    for alt,ust in renk_araliklari:
        alt_np = np.array(alt, dtype=np.uint8)
        ust_np = np.array(ust, dtype=np.uint8)
        maske = cv2.inRange(hsv,alt_np, ust_np)
        toplam_maske = cv2.bitwise_or(toplam_maske, maske)


    sonuc = cv2.bitwise_and(goruntu, goruntu, mask=toplam_maske )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(toplam_maske, cmap='gray')
    plt.title("Toplam Maske")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
    plt.title("Maskelenmiş Görüntü")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return toplam_maske, sonuc

goruntu = test_goruntu_olustur()
renkler = [
    ([20,100,100],[30,255,255]),
    ([0,100,100],[10,255,255]),
    ([160,100,100],[179,255,255])
]
maske, sonuc = coklu_renk_maskeleme(goruntu, renkler)


