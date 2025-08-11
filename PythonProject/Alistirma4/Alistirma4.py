import cv2
import numpy as np
import matplotlib.pyplot as plt

def coklu_renk_maskeleme(goruntu, renk_araliklari):

    hsv_goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)

    birlesik_maske = np.zeros_like(hsv_goruntu[:, :, 0])

    for (alt_sinir, ust_sinir) in renk_araliklari:
        alt = np.array(alt_sinir)
        ust = np.array(ust_sinir)
        maske = cv2.inRange(hsv_goruntu, alt, ust)
        birlesik_maske = cv2.bitwise_or(birlesik_maske, maske)

        sonuc = cv2.bitwise_and(goruntu,goruntu,mask=birlesik_maske)

    return sonuc, birlesik_maske


goruntu = cv2.imread('lena.jpg')

renk_araliklari = [
    ([0, 100, 100], [10, 255, 255]), #sarı
    ([100, 100, 100], [130, 255, 255]) #mavi
]

sonuc, maske = coklu_renk_maskeleme(goruntu,renk_araliklari)

cv2.imshow('sonuc', sonuc)
cv2.imshow('birleşik maske', maske)
cv2.imshow('orijinal', goruntu)
cv2.waitKey(0)
cv2.destroyAllWindows()