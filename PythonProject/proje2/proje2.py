'''
Proje 2: Belge Tarayıcı
Kamera ile çekilen belge fotoğrafını düzelten sistem:

Kenar tespiti ile belge sınırlarını bulma
Perspektif düzeltme
Arka plan temizleme
Metin netleştirme
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


def _kose_noktalari_sirala(noktalar: np.ndarray) -> np.ndarray:
    """Düzgün perspektif için 4 köşe noktasını [sol-üst, sağ-üst, sağ-alt, sol-alt] sırala."""
    pts = noktalar.reshape(4, 2).astype(np.float32)
    toplam = pts.sum(axis=1)
    fark = np.diff(pts, axis=1).reshape(-1)

    sirali = np.zeros((4, 2), dtype=np.float32)
    sirali[0] = pts[np.argmin(toplam)]   # sol-üst
    sirali[2] = pts[np.argmax(toplam)]   # sağ-alt
    sirali[1] = pts[np.argmin(fark)]     # sağ-üst
    sirali[3] = pts[np.argmax(fark)]     # sol-alt
    return sirali


def kenar_tespiti_ve_duzeltme(goruntu: np.ndarray):
    """
    Belgenin kenarlarını tespit eder ve perspektif düzeltmesi yapar.
    Daha kararlı sonuç için ölçek küçültme + morfoloji kullanır.
    """
    orj_h, orj_w = goruntu.shape[:2]
    hedef_yukseklik = 700
    oran = orj_h / float(hedef_yukseklik)
    kucuk = cv2.resize(goruntu, (int(orj_w / oran), hedef_yukseklik))

    gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)
    gri = cv2.bilateralFilter(gri, d=9, sigmaColor=75, sigmaSpace=75)

    v = np.median(gri)
    alt = int(max(0, (1.0 - 0.33) * v))
    ust = int(min(255, (1.0 + 0.33) * v))
    kenarlar = cv2.Canny(gri, alt, ust)

    cekirdek = np.ones((3, 3), np.uint8)
    kenarlar = cv2.dilate(kenarlar, cekirdek, iterations=1)
    kenarlar = cv2.erode(kenarlar, cekirdek, iterations=1)

    konturlar, _ = cv2.findContours(kenarlar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konturlar:
        return None, "Kontur bulunamadı"

    konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)[:10]
    belge_konturu = None
    for kontur in konturlar:
        cevre = cv2.arcLength(kontur, True)
        yaklasik = cv2.approxPolyDP(kontur, 0.02 * cevre, True)
        if len(yaklasik) == 4:
            belge_konturu = yaklasik
            break

    if belge_konturu is None:
        # Yine de en büyük dikdörtgensi kutuyu kullan (yedek plan)
        dik_kutu = cv2.minAreaRect(max(konturlar, key=cv2.contourArea))
        kutu_noktalari = cv2.boxPoints(dik_kutu)
        belge_konturu = kutu_noktalari.reshape(-1, 1, 2).astype(np.float32)

    siralanmis_kucuk = _kose_noktalari_sirala(belge_konturu.reshape(4, 2))
    siralanmis = (siralanmis_kucuk * oran).astype(np.float32)

    widthA = np.linalg.norm(siralanmis[2] - siralanmis[3])
    widthB = np.linalg.norm(siralanmis[1] - siralanmis[0])
    heightA = np.linalg.norm(siralanmis[1] - siralanmis[2])
    heightB = np.linalg.norm(siralanmis[0] - siralanmis[3])
    genislik = int(max(widthA, widthB))
    yukseklik = int(max(heightA, heightB))

    if genislik < 50 or yukseklik < 50:
        return None, "Belge çok küçük tespit edildi"

    hedef = np.array([
        [0, 0],
        [genislik - 1, 0],
        [genislik - 1, yukseklik - 1],
        [0, yukseklik - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(siralanmis, hedef)
    duzeltilmis = cv2.warpPerspective(goruntu, M, (genislik, yukseklik))

    return duzeltilmis, "Başarılı"


def arka_plan_temizleme(goruntu: np.ndarray) -> np.ndarray:
    """Arka planı temizleyip metni belirginleştir. Adaptif eşikleme kullanır."""
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    # Aydınlık değişimlerine dayanıklı adaptif eşikleme
    binary = cv2.adaptiveThreshold(
        gri, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 15
    )
    cekirdek = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cekirdek, iterations=1)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def metin_netlestirme(goruntu: np.ndarray) -> np.ndarray:
    """Metni hafifçe keskinleştir. Siyah-beyaz görüntülerde nazik davran."""
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    bulanık = cv2.GaussianBlur(gri, (0, 0), sigmaX=1.0)
    keskin = cv2.addWeighted(gri, 1.5, bulanık, -0.5, 0)
    return cv2.cvtColor(keskin, cv2.COLOR_GRAY2BGR)


def belge_tarayici(gorsel_yolu: str, kaydet: bool = False):
    """Ana belge tarayıcı fonksiyonu"""
    goruntu = cv2.imread(gorsel_yolu)
    if goruntu is None:
        print("Görüntü yüklenemedi!")
        return

    duzeltilmis, durum = kenar_tespiti_ve_duzeltme(goruntu)
    if durum != "Başarılı" or duzeltilmis is None:
        print(durum)
        return

    temizlenmis = arka_plan_temizleme(duzeltilmis)
    sonuc = metin_netlestirme(temizlenmis)

    plt.figure(figsize=(15, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Belge')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(duzeltilmis, cv2.COLOR_BGR2RGB))
    plt.title('Düzeltilmiş Belge')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(sonuc, cv2.COLOR_BGR2RGB))
    plt.title('Temizlenmiş ve Netleştirilmiş')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    if kaydet:
        cv2.imwrite('duzeltilmis_belge.jpg', sonuc)
        print("Düzeltilmiş belge kaydedildi: duzeltilmis_belge.jpg")

    return sonuc


if __name__ == "__main__":
    sonuc = belge_tarayici('ab.jpeg', kaydet=True)


