import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def goruntu_yukle_ve_goster():
    goruntu = cv2.imread("C:/Users/sudea/Downloads/lena (1).png")
    goruntu_rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    print(f"Gorutu boyutu: {goruntu.shape}")
    print(f"Veri tipi: {goruntu.dtype}")
    print(f" Min deger: {goruntu.min()} , Max deger: {goruntu.max()}") #parlaklık aralığını anlamak için.

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(goruntu_rgb)
    plt.title('Orjinal Goruntu')
    plt.axis('off')


    plt.subplot(1,2,2)
    plt.hist(goruntu.ravel(), bins=256, range=[0,256])
    plt.title('Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.tight_layout()
    plt.show()

    return goruntu

def test_goruntu_olustur():

    goruntu = np.zeros((512, 512, 3), dtype=np.uint8) #bgr formatinda su an
    goruntu[:, :] =[255, 220, 177] #arkaplani ten rengi yaptik
    cv2.ellipse(goruntu,(256,150), (180, 80), 0, 180, 360, (30, 200, 255), -1)
    # -1 ici dolu ciz
    cv2.circle(goruntu, (256, 280), 100, (255, 200, 150), -1)
    cv2.circle(goruntu, (220, 260), 15, (50, 50, 50), -1)
    cv2.circle(goruntu, (290, 260), 15, (50, 50, 50), -1)
    cv2.ellipse(goruntu, (256, 300), (50, 30), 0, 0, 180, (100, 50, 50), 3)

    return goruntu
goruntu_yukle_ve_goster()
test_goruntu_olustur()









































