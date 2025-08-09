import numpy as np
import cv2
import matplotlib.pyplot as plt

def renk_tespit(goruntu):

    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)


    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    kirmizi_maske = cv2.bitwise_or(mask1, mask2)


    sonuc = cv2.bitwise_and(goruntu, goruntu, mask=kirmizi_maske)


    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Orijinal Görüntü")
    plt.imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Maske")
    plt.imshow(kirmizi_maske, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Sadece Kırmızı")
    plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()



goruntu = cv2.imread("C:/Users/sudea/Downloads/lena (1).png")
renk_tespit(goruntu)
