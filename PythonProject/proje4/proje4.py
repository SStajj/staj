import os
from typing import List, Tuple

import cv2
import numpy as np

''' olmadı bu '''

def harris_koseler(gri: np.ndarray, blok: int = 2, ksize: int = 3, k: float = 0.04,
                   esik_orani: float = 0.01, maksimum_nokta: int = 1000) -> List[cv2.KeyPoint]:
    """
    1) Özellik noktaları: Harris corner detector ile köşeleri bul, en güçlüleri seç.
    """
    harris = cv2.cornerHarris(gri.astype(np.float32), blockSize=blok, ksize=ksize, k=k)
    harris = cv2.dilate(harris, None)
    esik = esik_orani * harris.max()

    ylar, xlar = np.where(harris > esik)
    puanlar = harris[ylar, xlar]
    sirali = np.argsort(-puanlar)[:maksimum_nokta]

    keypoints: List[cv2.KeyPoint] = []
    for idx in sirali:
        x = float(xlar[idx])
        y = float(ylar[idx])
        keypoints.append(cv2.KeyPoint(x, y, 3))
    return keypoints


def ozellik_eslestir(img1: np.ndarray, img2: np.ndarray) -> Tuple[List[cv2.DMatch], List[cv2.KeyPoint], List[cv2.KeyPoint]]:
    """
    2) Görüntü eşleştirme: Harris noktalarından ORB tanımlayıcı çıkar ve eşleştir (Lowe oran testi).
    """
    gri1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gri2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1 = harris_koseler(gri1)
    kp2 = harris_koseler(gri2)

    orb = cv2.ORB_create(nfeatures=max(len(kp1), len(kp2)) or 500)
    kp1, des1 = orb.compute(gri1, kp1)
    kp2, des2 = orb.compute(gri2, kp2)

    if des1 is None or des2 is None:
        return [], kp1 or [], kp2 or []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    ham = bf.knnMatch(des1, des2, k=2)

    iyi_eslesmeler: List[cv2.DMatch] = []
    for cift in ham:
        if len(cift) < 2:
            continue
        m, n = cift
        if m.distance < 0.75 * n.distance:
            iyi_eslesmeler.append(m)

    iyi_eslesmeler = sorted(iyi_eslesmeler, key=lambda m: m.distance)
    return iyi_eslesmeler, kp1, kp2


def homografi_hesapla(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                      eslesmeler: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
    """
    3) Homografi: RANSAC ile H 3x3 hesapla.
    """
    if len(eslesmeler) < 4:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in eslesmeler]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in eslesmeler]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H, mask


def birlestir_panorama(sol: np.ndarray, sag: np.ndarray, H_sag2sol: np.ndarray) -> np.ndarray:
    """
    4) Birleştirme: sağ görüntüyü sol düzleme warp et ve mesafe-dönüşümü tabanlı feather blending uygula.
    """
    h1, w1 = sol.shape[:2]
    h2, w2 = sag.shape[:2]

    # Tuval boyutunu belirle
    koseler_sag = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    koseler_sag_warp = cv2.perspectiveTransform(koseler_sag, H_sag2sol)
    koseler_sol = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

    tum_koseler = np.vstack((koseler_sol, koseler_sag_warp))
    [xmin, ymin] = np.floor(tum_koseler.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(tum_koseler.max(axis=0).ravel()).astype(int)

    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    pano_w, pano_h = xmax - xmin, ymax - ymin

    warp_sag = cv2.warpPerspective(sag, T @ H_sag2sol, (pano_w, pano_h))

    # Sol görüntüyü pano düzlemine yerleştir
    pano_sol = np.zeros_like(warp_sag)
    pano_sol[ty:ty + h1, tx:tx + w1] = sol

    # İkili maskeler
    mask_sol = np.zeros((pano_h, pano_w), dtype=np.uint8)
    mask_sol[ty:ty + h1, tx:tx + w1] = (cv2.cvtColor(sol, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255

    mask_sag = (warp_sag.sum(axis=2) > 0).astype(np.uint8) * 255

    # Mesafe dönüşümü (kenardan uzaklık → daha yüksek ağırlık)
    dist_sol = cv2.distanceTransform(mask_sol, cv2.DIST_L2, 3)
    dist_sag = cv2.distanceTransform(mask_sag, cv2.DIST_L2, 3)

    # Ağırlık haritalarını yumuşat ve normalleştir
    dist_sol = cv2.GaussianBlur(dist_sol, (15, 15), 0)
    dist_sag = cv2.GaussianBlur(dist_sag, (15, 15), 0)

    toplam = dist_sol + dist_sag + 1e-6
    w_sol = (dist_sol / toplam).astype(np.float32)
    w_sag = (dist_sag / toplam).astype(np.float32)

    # Dış bölgelerde (tek maske) ağırlıkları 0/1'e sabitle
    sadece_sol = (mask_sol > 0) & (mask_sag == 0)
    sadece_sag = (mask_sag > 0) & (mask_sol == 0)
    w_sol[sadece_sol] = 1.0
    w_sag[sadece_sol] = 0.0
    w_sol[sadece_sag] = 0.0
    w_sag[sadece_sag] = 1.0

    # Karıştırma
    pano = (
        pano_sol.astype(np.float32) * w_sol[..., None] +
        warp_sag.astype(np.float32) * w_sag[..., None]
    ).astype(np.uint8)

    return pano


def ornek_calistir() -> str:
    """
    Okunabilir, kısa demo: aynı görüntüden kaydırılmış ikincisini üret ve panorama oluştur.
    """
    # Lena yolunu belirle (proje kökünde varsayılır)
    proje_kok = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    lena_yol = os.path.join(proje_kok, 'dnm3.jpeg')
    img = cv2.imread('dnm4.jpeg')
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {'yuz3.png'}")

    # Sentetik sağ görüntü (sağa kaydırma)
    M = np.float32([[1, 0, 80], [0, 1, 0]])
    img_sag = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    eslesmeler, kp1, kp2 = ozellik_eslestir(img, img_sag)
    H, mask = homografi_hesapla(kp1, kp2, eslesmeler)
    if H is None:
        raise RuntimeError("Yeterli eşleşme bulunamadı; homografi hesaplanamadı.")

    pano = birlestir_panorama(img, img_sag, H)

    cikti_yol = os.path.join(os.path.dirname(__file__), 'P4_Sonuc.png')
    cv2.imwrite(cikti_yol, pano)
    return cikti_yol


if __name__ == '__main__':
    yol = ornek_calistir()
    print(f"Panorama kaydedildi: {yol}")


