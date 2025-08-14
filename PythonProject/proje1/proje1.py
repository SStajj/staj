import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
Proje 1: Otomatik Renk Düzeltme
Bir görüntünün renk dengesini otomatik düzelten sistem yazın:

Beyaz dengesi ayarlama
Histogram eşitleme
Kontrast ve parlaklık optimizasyonu
'''


def beyaz_dengesi(image):
    """
    Gri Dünya algoritması ile otomatik beyaz dengesi ayarlama
    """
    # Görüntüyü float32'ye çevir
    img = image.astype(np.float32) / 255.0

    # Her kanalın ortalamasını hesapla
    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])

    # Gri dünya varsayımı: ortalama R=G=B olmalı
    avg_gray = (avg_r + avg_g + avg_b) / 3.0

    # Ölçek faktörlerini hesapla
    scale_r = avg_gray / avg_r
    scale_g = avg_gray / avg_g
    scale_b = avg_gray / avg_b

    # Ölçek faktörlerini uygula
    img[:, :, 0] = np.clip(img[:, :, 0] * scale_r, 0, 1)
    img[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 1)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_b, 0, 1)

    return (img * 255).astype(np.uint8)


def histogram_Esitleme(image):
    """
    Kontrast sınırlı adaptif histogram eşitleme (CLAHE)
    """
    if len(image.shape) == 3:
        # Renkli görüntü için HSV uzayında Value kanalında eşitleme
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # CLAHE uygula
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)

        # Kanalları birleştir
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        # Gri tonlamalı görüntü
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def optimize_contrast_brightness(image):
    """
    Otomatik kontrast ve parlaklık ayarlama
    """
    # Gri tonlamalı görüntüye çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu thresholding ile optimal eşik değeri bul
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Parlaklık ve kontrast için alpha ve beta değerleri hesapla
    alpha = 1.5  # Kontrast (1.0-3.0)
    beta = -50  # Parlaklık (-50 ile +50)

    # Kontrast ve parlaklık ayarla
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted


def auto_color_correction(image_path):
    """
    Otomatik renk düzeltme işlemlerini uygular ve sonuçları gösterir
    """
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü yüklenemedi!")
        return

    # 1. Beyaz dengesi ayarla
    white_balanced = beyaz_dengesi(image)

    # 2. Histogram eşitleme
    histogram_equalized = histogram_Esitleme(white_balanced)

    # 3. Kontrast ve parlaklık optimizasyonu
    final_image = optimize_contrast_brightness(histogram_equalized)

    # Sonuçları görselleştir
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Görüntü')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(white_balanced, cv2.COLOR_BGR2RGB))
    plt.title('Beyaz Dengesi Ayarlanmış')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(histogram_equalized, cv2.COLOR_BGR2RGB))
    plt.title('Histogram Eşitlenmiş')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title('Kontrast Optimize Edilmiş (Sonuç)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return final_image

# Örnek kullanım
corrected_image = auto_color_correction('lena.jpg')