import os
import cv2
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


def gurultu_gider(resim, yontem= "gaussian", kernel_size = (3,3)):

    if yontem == "gaussian":
        claned = cv2.GaussianBlur(resim,kernel_size,0)
    elif yontem == "median":
        claned = cv2.medianBlur(resim,kernel_size[0])
    elif yontem == "bilateral":
        claned = cv2.bilateralFilter(resim,9,75,75)
    else:
        raise ValueError("Geçersiz Yöntem")

    return claned


def egrilik_duzelt(image, max_angle=5):

    # 1. Eğrilik açısını hesapla
    coords = np.column_stack(np.where(image > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # 2. Açıyı normalize et (-90° ile 90° arası)
    angle = angle - 90 if angle > 45 else angle

    # 3. Eğiklik kritik değeri aşmıyorsa orijinali döndür
    if abs(angle) <= max_angle:
        return image.copy()

    # 4. Döndürme işlemi
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed


def karakter_segmentasyonu(ikili_inv: np.ndarray, girdi_bgr: np.ndarray) -> np.ndarray:

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temiz = cv2.morphologyEx(ikili_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    temiz = cv2.morphologyEx(temiz, cv2.MORPH_CLOSE, kernel, iterations=1)

    sayi, etiketler, istatistik, _ = cv2.connectedComponentsWithStats(temiz, connectivity=8)
    h, w = temiz.shape
    goruntu = girdi_bgr.copy()
    goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    goruntu = cv2.cvtColor(goruntu, cv2.COLOR_RGB2BGR)

    goruntu_cizim = girdi_bgr.copy()
    tum_alan = h * w
    for i in range(1, sayi):
        x, y, bw, bh, alan = istatistik[i]
        if alan < 60 or alan > tum_alan * 0.5:
            continue
        cv2.rectangle(goruntu_cizim, (x, y), (x + bw, y + bh), (0, 200, 0), 1)

    return goruntu_cizim

def main():
    resim = cv2.imread('metin.jpg', cv2.IMREAD_GRAYSCALE)

    secim = input("Gürültü giderme yöntemi seçin (gaussian, median, bilateral): ").lower()

    try:
        cekirdek_str = input("Kernel size giriniz (örn: 3,3): ")
        cekirdek = tuple(map(int, cekirdek_str.split(',')))
        if len(cekirdek) != 2:
            raise ValueError("Kernel size iki elemanlı olmalı (örneğin: 3,3)")
    except ValueError as e:
        print(f"Hata: Geçersiz kernel size! {e}")
        return

    gurultusuz= gurultu_gider(resim,secim,cekirdek)

    _, binary = cv2.threshold(gurultusuz, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    deskew = egrilik_duzelt(binary)

    segmentli_goruntu = karakter_segmentasyonu(deskew, binary)

    # Görselleştirme
    # 1. Gürültüsüz Görüntü
    plt.subplot(221), plt.imshow(gurultusuz, cmap='gray' if len(gurultusuz.shape) == 2 else None)
    plt.title('1. Gürültüsüz Görüntü'), plt.axis('off')

    # 2. Binary Görüntü
    plt.subplot(222), plt.imshow(binary, cmap='gray')
    plt.title('2. Otsu Binarizasyon'), plt.axis('off')

    # 3. Deskew Edilmiş Görüntü
    plt.subplot(223), plt.imshow(deskew, cmap='gray')
    plt.title('3. Deskew (Düzeltilmiş)'), plt.axis('off')

    # 4. Segmentasyon Sonucu
    plt.subplot(224), plt.imshow(segmentli_goruntu)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()