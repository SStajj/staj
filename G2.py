import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# RGB renk karışımını anlamak
def rgb_karisimi_goster():
    # Boş bir görüntü oluştur
    goruntu = np.zeros((300, 400, 3), dtype=np.uint8)

    # Farklı bölgelere farklı renkler
    goruntu[0:100, 0:100] = [255, 0, 0]  # Saf Kırmızı
    goruntu[0:100, 100:200] = [0, 255, 0]  # Saf Yeşil
    goruntu[0:100, 200:300] = [0, 0, 255]  # Saf Mavi
    goruntu[0:100, 300:400] = [255, 255, 255]  # Beyaz

    goruntu[100:200, 0:100] = [255, 255, 0]  # Sarı (R+G)
    goruntu[100:200, 100:200] = [255, 0, 255]  # Magenta (R+B)
    goruntu[100:200, 200:300] = [0, 255, 255]  # Cyan (G+B)
    goruntu[100:200, 300:400] = [128, 128, 128]  # Gri

    # OpenCV BGR formatında çalışır, RGB'ye çevir
    goruntu_rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(goruntu_rgb)
    plt.title('RGB Renk Karışımları')
    plt.axis('off')
    plt.show()

    return goruntu


# Fonksiyonu çalıştır
renk_ornegi = rgb_karisimi_goster()