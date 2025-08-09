import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from G4 import test_goruntu_olustur


def hsv_maskeleme():
    goruntu = test_goruntu_olustur()
    hsv =cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)

    alt_sari = np.array([20,100,100])
    ust_sari = np.array([30,255,255])

    maske = cv2.inRange(hsv, alt_sari,ust_sari)

    sonuc = cv2.bitwise_and(goruntu, goruntu,mask=maske)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(goruntu,cv2.COLOR_BGR2RGB))
    plt.title('Orjinal')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
    plt.title('Sarı Maskeleme')
    plt.axis('off')

    plt.show()

    return maske


sari_maske = hsv_maskeleme()