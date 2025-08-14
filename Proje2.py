import cv2
import numpy as np
import matplotlib.pyplot as plt

'''Proje 2: Belge Tarayıcı
Kamera ile çekilen belge fotoğrafını düzelten sistem:

Kenar tespiti ile belge sınırlarını bulma
Perspektif düzeltme
Arka plan temizleme
Metin netleştirme'''
# ----------------------
# Köşe Sıralama
# ----------------------
def sirala_koseler(koseler):
    koseler = np.array(koseler).reshape(-1, 2).astype(np.float32)
    merkez = np.mean(koseler, axis=0)
    acilar = [np.arctan2(y - merkez[1], x - merkez[0]) for x, y in koseler]
    siralama = np.argsort(acilar)
    koseler = koseler[siralama]
    toplam = np.sum(koseler, axis=1)
    sol_ust_idx = np.argmin(toplam)
    rect = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        rect[i] = koseler[(sol_ust_idx + i) % 4]
    return rect

# ----------------------
# Gelişmiş Kenar Tespiti
# ----------------------
def gelismis_kenar_tespiti(goruntu):
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gri = clahe.apply(gri)
    blur1 = cv2.GaussianBlur(gri, (5, 5), 0)
    blur2 = cv2.medianBlur(gri, 5)
    blur3 = cv2.bilateralFilter(gri, 9, 75, 75)
    kenar1 = cv2.Canny(blur1, 30, 80)
    kenar2 = cv2.Canny(blur2, 50, 120)
    kenar3 = cv2.Canny(blur3, 40, 100)
    kenar4 = cv2.Canny(gri, 60, 150)
    combined = cv2.bitwise_or(cv2.bitwise_or(kenar1, kenar2),
                              cv2.bitwise_or(kenar3, kenar4))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.dilate(combined, kernel, iterations=2)
    return combined

# ----------------------
# Geometrik Kontrol
# ----------------------
def geometrik_kontrol(koseler, w, h):
    kenarlar = [np.linalg.norm(koseler[i] - koseler[(i + 1) % 4]) for i in range(4)]
    if min(kenarlar) < 30: return False
    if max(kenarlar) > min(w, h) * 0.9: return False
    alan = cv2.contourArea(koseler.astype(np.int32))
    if alan < w * h * 0.01: return False
    for i in range(4):
        p1, p2, p3 = koseler[i], koseler[(i + 1) % 4], koseler[(i + 2) % 4]
        v1, v2 = p1 - p2, p3 - p2
        cos_aci = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        aci = np.degrees(np.arccos(np.clip(cos_aci, -1, 1)))
        if aci < 30 or aci > 150: return False
    return True

# ----------------------
# Köşe Kalite Puanı
# ----------------------
def kalite_puani(koseler, alan, w, h):
    puan = 0
    ideal_alan = w * h * 0.3
    alan_farki = abs(alan - ideal_alan) / ideal_alan
    puan += max(0, 1 - alan_farki) * 50
    kenarlar = [np.linalg.norm(koseler[i] - koseler[(i + 1) % 4]) for i in range(4)]
    oran = max(kenarlar) / min(kenarlar)
    if oran < 3:
        puan += (3 - oran) * 10
    merkez_img = np.array([w / 2, h / 2])
    merkez_kose = np.mean(koseler, axis=0)
    merkez_uzaklik = np.linalg.norm(merkez_kose - merkez_img)
    puan += max(0, 1 - merkez_uzaklik / (min(w, h) * 0.3)) * 20
    return puan

# ----------------------
# Üst Yüzey Köşeleri
# ----------------------
def ust_yuzey_koseleri(goruntu):
    h, w = goruntu.shape[:2]
    kenarlar = gelismis_kenar_tespiti(goruntu)
    konturlar, _ = cv2.findContours(kenarlar, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)
    min_alan, max_alan = w * h * 0.01, w * h * 0.95
    en_iyi_koseler, en_iyi_puan = None, -1
    for kontur in konturlar[:20]:
        alan = cv2.contourArea(kontur)
        if alan < min_alan or alan > max_alan: continue
        peri = cv2.arcLength(kontur, True)
        approx = cv2.approxPolyDP(kontur, 0.02 * peri, True)
        if len(approx) == 4:
            koseler = approx.reshape(4, 2)
            if geometrik_kontrol(koseler, w, h):
                puan = kalite_puani(koseler, alan, w, h)
                if puan > en_iyi_puan:
                    en_iyi_puan, en_iyi_koseler = puan, koseler
    if en_iyi_koseler is not None:
        return sirala_koseler(en_iyi_koseler), kenarlar
    return None, kenarlar

# ----------------------
# Perspektif Düzeltme + Dik ve Doğru Yön
# ----------------------
def perspektif_duzelt_dikey_duz_yon(goruntu, koseler):
    koseler = sirala_koseler(koseler)
    ust = np.linalg.norm(koseler[0] - koseler[1])
    sag = np.linalg.norm(koseler[1] - koseler[2])
    alt = np.linalg.norm(koseler[2] - koseler[3])
    sol = np.linalg.norm(koseler[3] - koseler[0])
    genislik, yukseklik = int(max(ust, alt)), int(max(sol, sag))
    hedef = np.array([[0, 0], [genislik - 1, 0],
                      [genislik - 1, yukseklik - 1], [0, yukseklik - 1]],
                     dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(koseler.astype(np.float32), hedef)
    duzeltilmis = cv2.warpPerspective(goruntu, matrix, (genislik, yukseklik))

    # Yan yatıksa dik çevir
    if duzeltilmis.shape[1] > duzeltilmis.shape[0]:
        duzeltilmis = cv2.rotate(duzeltilmis, cv2.ROTATE_90_CLOCKWISE)

    # 180° terslik kontrolü
    h, w = duzeltilmis.shape[:2]
    ust_yarisi = duzeltilmis[:h//2, :]
    alt_yarisi = duzeltilmis[h//2:, :]
    if np.mean(ust_yarisi) < np.mean(alt_yarisi):
        duzeltilmis = cv2.rotate(duzeltilmis, cv2.ROTATE_180)

    return duzeltilmis

# ----------------------
# Arka Plan Temizleme + Metin Netleştirme
# ----------------------
def arka_plan_temizle(goruntu):
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    gri = cv2.equalizeHist(gri)  # kontrast artırma
    temiz = cv2.adaptiveThreshold(gri, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    return temiz

def metin_netlestir(goruntu):
    kernel = np.ones((1, 1), np.uint8)
    net = cv2.morphologyEx(goruntu, cv2.MORPH_CLOSE, kernel)
    return net

# ----------------------
# Ana Çalışma
# ----------------------
def main():
    goruntu = cv2.imread("../RESIMLER/proje222.jpg")
    koseler, kenarlar = ust_yuzey_koseleri(goruntu)
    if koseler is not None:
        duzeltilmis = perspektif_duzelt_dikey_duz_yon(goruntu, koseler)
        temiz = arka_plan_temizle(duzeltilmis)
        net = metin_netlestir(temiz)
    else:
        print("Köşe bulunamadı!")
        duzeltilmis, temiz, net = None, None, None

    # Görselleştirme
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal")
    plt.subplot(2, 3, 2)
    plt.imshow(kenarlar, cmap='gray')
    plt.title("Kenarlar")
    if duzeltilmis is not None:
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(duzeltilmis, cv2.COLOR_BGR2RGB))
        plt.title("Dik ve Doğru Yön")
        plt.subplot(2, 3, 4)
        plt.imshow(temiz, cmap='gray')
        plt.title("Arka Plan Temiz")
        plt.subplot(2, 3, 5)
        plt.imshow(net, cmap='gray')
        plt.title("Netleştirilmiş (OCR için Hazır)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
