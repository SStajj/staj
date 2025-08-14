import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

'''Proje 1: Otomatik Renk Düzeltme
Bir görüntünün renk dengesini otomatik düzelten sistem yazın:

Beyaz dengesi ayarlama
Histogram eşitleme
Kontrast ve parlaklık optimizasyonu'''

def white_balance(img):
    b, g, r = cv2.split(img)
    b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)

    b = np.clip(b * (128 / b_avg), 0, 255).astype(np.uint8)
    g = np.clip(g * (128 / g_avg), 0, 255).astype(np.uint8)
    r = np.clip(r * (128 / r_avg), 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])


def hist_esitleme(img):
    # HSV renk uzayında sadece Value (parlaklık) kanalında eşitleme
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def kontrast_artir(img, alpha=1.5):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def renk_duzelt(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Hata: Görüntü yüklenemedi!")
        return None

    wb = white_balance(img)
    he = hist_esitleme(wb)
    final = kontrast_artir(he)

    return img, wb, he, final


def main(img_path):
    sonuclar = renk_duzelt(img_path)
    if sonuclar is None:
        return

    img, wb, he, final = sonuclar

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(wb, cv2.COLOR_BGR2RGB))
    plt.title("Beyaz Dengesi")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(he, cv2.COLOR_BGR2RGB))
    plt.title("Histogram Eşitleme")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title("Son Hali (Kontrast)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = "../RESIMLER/denemeal6.jpg"
    if Path(img_path).exists():
        main(img_path)
    else:
        print("Dosya bulunamadı!")