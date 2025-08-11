
import cv2
import numpy as np
import matplotlib.pyplot as plt

def basit_clahe(goruntu, blok_boyutu = 8):

    if len(goruntu.shape)==3:
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)

    yukseklik,genislik = goruntu.shape
    sonuc = np.zeros_like(goruntu)

    #simdi bloklara bolucem
    for i in range(0,yukseklik,blok_boyutu):
        for j in range(0, genislik, blok_boyutu):

            i_son = min(i + blok_boyutu,yukseklik)
            j_son = min(j + blok_boyutu, genislik)


            blok = goruntu[i:i_son, j:j_son]
            hist = np.bincount(blok.ravel(),minlength=256)

            esitlenmis_blok = histogram_esitle_blok(blok)

            sonuc[i:i_son, j:j_son] = esitlenmis_blok

    return sonuc

def histogram_esitle_blok(blok):
    return cv2.equalizeHist(blok)
img = cv2.imread("C:/Users/sudea/Downloads/lena (1).png")           # kendi yolunu yaz
out = basit_clahe(img, blok_boyutu=8)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.title("Orijinal (Gray)")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap="gray"); plt.axis("off")
plt.subplot(1,2,2); plt.title("Basit CLAHE (blok=8)")
plt.imshow(out, cmap="gray"); plt.axis("off")
plt.tight_layout(); plt.show()




