import cv2
import numpy as np
import matplotlib.pyplot as plt

'''HSV kullanarak belirli bir rengi tespit eden fonksiyon yazın:'''

def renk_tespit(goruntu, alt_sinir_hsv, ust_sinir_hsv):
    # Görüntüyü HSV renk uzayına dönüştür
    hsv_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)

    # HSV değerlerini numpy array'e çevir
    alt_sinir = np.array(alt_sinir_hsv)
    ust_sinir = np.array(ust_sinir_hsv)

    # Belirtilen HSV aralığındaki renkleri maskele
    mask = cv2.inRange(hsv_goruntu, alt_sinir, ust_sinir)

    return mask

image = cv2.imread('lena.jpg')

# Sarı renk için HSV aralıkları
alt_sinir = [0, 100, 100]
ust_sinir = [10, 255, 255]

# Renk tespiti yap
mask = renk_tespit(image, alt_sinir, ust_sinir)

# Orijinal görüntüde sadece sarı rengi göster
result = cv2.bitwise_and(image, image, mask=mask)

# Sonuçları göster
cv2.imshow('Orijinal', image)
cv2.imshow('Mask', mask)
cv2.imshow('Sonuc', result)
cv2.waitKey(0)
cv2.destroyAllWindows()