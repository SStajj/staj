import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from G4 import test_goruntu_olustur


def maskeleme_temelleri():
    goruntu = test_goruntu_olustur()

    def olustur_binary_maske(goruntu, esik_degeri):
        gri= cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY) #siyah beyaz gri tonlarina getir daha rahat
        maske = np.where(gri > esik_degeri, 255, 0).astype(np.uint8)
        return maske
    maske= olustur_binary_maske(goruntu,127)
    maskelenmis = cv2.bitwise_and(goruntu, goruntu, mask=maske)
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Orjinal')
    axes[0].axis('off')

    axes[1].imshow(maske, cmap='gray')
    axes[1].set_title('Binary Maske')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(maskelenmis,cv2.COLOR_BGR2RGB))
    axes[2].set_title('Maskelenmis Goruntu')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return maske,maskelenmis
maske,maskelenmis = maskeleme_temelleri()



