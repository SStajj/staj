import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#histogram esitleme
#white balance +
#kontrast ayarlama
#ana islemler

def white_balance(goruntu):
    b,g,r = cv2.split(goruntu)
    b_avg, g_avg, r_avg = np.mean(b), np.mean(g),np.mean(r)

    b = np.clip(b * (128 / b_avg), 0,255).astype(np.uint8)
    g = np.clip(g * (128 / g_avg), 0,255).astype(np.uint8)
    r = np.clip(r * (128 / r_avg), 0,255).astype(np.uint8)

    return cv2.merge([b,g,r])
def hist_esitleme(goruntu):
    if len(goruntu.shape)==3:
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    else:
        gri = goruntu.copy()
    esitlenmis = cv2.equalizeHist(gri)
    return esitlenmis
def kontrat_ayarlama(goruntu, alpha = 1.5):
    return cv2.convertScaleAbs(goruntu, alpha=alpha, beta=0)
def renk_duzelt(goruntu_yolu):
    img = cv2.imread(goruntu_yolu)
    if img is None:
        print("Hata: Goruntu yuklenmedi!")
        return None
    wb = white_balance(img)
    he = hist_esitleme(wb)  # gösterim için
    final = kontrat_ayarlama(wb)  # senin akışın: kontrast WB üstüne
    return img, wb, he, final


def main(goruntu_yolu):
    sonuclar = renk_duzelt(goruntu_yolu)
    if sonuclar is None:
        return
    img, wb, he, final = sonuclar

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1);
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    plt.title("Orijinal");
    plt.axis('off')
    plt.subplot(1, 4, 2);
    plt.imshow(cv2.cvtColor(wb, cv2.COLOR_BGR2RGB));
    plt.title("Beyaz Dengesi");
    plt.axis('off')
    plt.subplot(1, 4, 3);
    plt.imshow(he, cmap='gray');
    plt.title("Histogram Eşitleme");
    plt.axis('off')
    plt.subplot(1, 4, 4);
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB));
    plt.title("Son Hali (Kontrast)");
    plt.axis('off')
    plt.tight_layout();
    plt.show()
    return img, wb, he, final
if __name__ == "__main__":
    img_path = "lena.png"
    if Path(img_path).exists(): main(img_path)
    else: print("Dosya bulunamadı!")
















