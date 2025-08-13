import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#hsv cevirme ten rengi testpiti
#bulaniklastirma(Gaussionblur)
#maske temizleme

def ten_rengi(goruntu):
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)

    alt_sinir = np.array([0, 60, 80])
    ust_sinir = np.array([25, 150, 255])  # Fixed np.astype to np.array

    mask = cv2.inRange(hsv, alt_sinir, ust_sinir)
    return mask

def maske_temizleme(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    temiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    temiz = cv2.morphologyEx(temiz, cv2.MORPH_CLOSE, kernel)

    return temiz

def bulaniklastirma_basit(goruntu, mask):
    bulanik = cv2.GaussianBlur(goruntu, (21, 21), 0)
    mask_normalize = mask.astype(np.float32)/ 255.0

    sonuc = goruntu.astype(np.float32)
    bulanik = bulanik.astype(np.float32)

    # Fixed the blending operation
    sonuc = goruntu * (1 - mask_normalize[:, :, np.newaxis]) + bulanik * mask_normalize[:, :, np.newaxis]

    return sonuc.astype(np.uint8)

def main():
    goruntu = cv2.imread("proje3gorsel.jpg")
    if goruntu is None:
        print("HATA: lena.png dosyası bulunamadı!")
        print("Lütfen aynı klasöre bir yüz fotoğrafı koyun")
        return

    print("📸 Görüntü yüklendi!")
    print(f"   Boyut: {goruntu.shape[1]}x{goruntu.shape[0]} piksel")

    # Adım adım işle
    print("\n🔍 ADIM 1: Ten rengi tespiti...")
    ten_mask = ten_rengi(goruntu)  # Fixed function name to match definition

    print("🧹 ADIM 2: Maskeyi temizleme...")
    temiz_mask = maske_temizleme(ten_mask)

    print("💫 ADIM 3: Bulanıklaştırma...")
    sonuc = bulaniklastirma_basit(goruntu, temiz_mask)

    # Sonuçları göster
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    plt.title("1. Orijinal Fotoğraf")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(ten_mask, cmap='gray')
    plt.title("2. Ten Rengi Maskesi\n(Beyaz=ten, Siyah=değil)")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(temiz_mask, cmap='gray')
    plt.title("3. Temizlenmiş Maske")
    plt.axis('off')

    # Sadece ten bölgelerini göster
    sadece_ten = cv2.bitwise_and(goruntu, goruntu, mask=temiz_mask)
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(sadece_ten, cv2.COLOR_BGR2RGB))
    plt.title("4. Tespit Edilen Ten Bölgeleri")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
    plt.title("5. ✨ SONUÇ ✨\nBulanık Yüzler")
    plt.axis('off')

    # Karşılaştırma
    plt.subplot(2, 3, 6)
    karsilastirma = np.hstack((goruntu[:, :goruntu.shape[1] // 2],
                               sonuc[:, sonuc.shape[1] // 2:]))
    plt.imshow(cv2.cvtColor(karsilastirma, cv2.COLOR_BGR2RGB))
    plt.title("6. Öncesi ↔️ Sonrası")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()