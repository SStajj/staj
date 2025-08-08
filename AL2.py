import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_kanallari_ayir(goruntu_yolu):

    bgr = cv2.imread(goruntu_yolu)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    R =rgb[:,:,0]
    G =rgb[:,:,1]
    B =rgb[:,:,2]


    plt.figure(figsize =(15, 5))

    plt.subplot(1,3,1)
    plt.imshow(R , cmap ='Reds')
    plt.title('Kirmizi kanal')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(G, cmap='Greens')
    plt.title('Yesil kanal')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(B, cmap='Blues')
    plt.title('Mavi kanal')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return rgb

rgb_kanallari_ayir("C:/Users/sudea/Downloads/lena (1).png")