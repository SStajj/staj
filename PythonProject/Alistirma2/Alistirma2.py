import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Alıştırma 2: RGB Kanal Ayrıştırma
Verilen bir renkli görüntüyü R, G, B kanallarına ayıran ve her kanalı ayrı ayrı gösteren bir fonksiyon yazın.

def rgb_kanallari_ayir(goruntu_yolu):
    """
    Görüntüyü RGB kanallarına ayırır ve görselleştirir
    """
    # Kodunuzu buraya yazın
    pass
'''

def rgb_kanallari_ayir(goruntu_yolu):

    # Görüntüyü oku (BGR formatında)
    goruntu = cv2.imread(goruntu_yolu)

    # OpenCV BGR formatını RGB'ye çevir (matplotlib için)
    goruntu_rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)

    # Kanalları ayır
    r_kanal = goruntu_rgb.copy()
    r_kanal[:, :, 1] = 0  # Yeşil kanalı sıfırla
    r_kanal[:, :, 2] = 0  # Mavi kanalı sıfırla

    g_kanal = goruntu_rgb.copy()
    g_kanal[:, :, 0] = 0  # Kırmızı kanalı sıfırla
    g_kanal[:, :, 2] = 0  # Mavi kanalı sıfırla

    b_kanal = goruntu_rgb.copy()
    b_kanal[:, :, 0] = 0  # Kırmızı kanalı sıfırla
    b_kanal[:, :, 1] = 0  # Yeşil kanalı sıfırla

    # Görselleştirme
    plt.figure(figsize=(12, 8))

    # Orijinal görüntü
    plt.subplot(2, 2, 1)
    plt.imshow(goruntu_rgb)
    plt.title('Orijinal Görüntü')
    plt.axis('off')

    # R kanalı
    plt.subplot(2, 2, 2)
    plt.imshow(r_kanal)
    plt.title('Kırmızı (R) Kanalı')
    plt.axis('off')

    # G kanalı
    plt.subplot(2, 2, 3)
    plt.imshow(g_kanal)
    plt.title('Yeşil (G) Kanalı')
    plt.axis('off')

    # B kanalı
    plt.subplot(2, 2, 4)
    plt.imshow(b_kanal)
    plt.title('Mavi (B) Kanalı')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


rgb_kanallari_ayir('lena.jpg')