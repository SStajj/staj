import cv2
import numpy as np
import matplotlib.pyplot as plt


def _ten_maskesi_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """HSV renk uzayında ten rengi maskesi (0/255)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 60], dtype=np.uint8)
    upper = np.array([25, 180, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _yuz_maskesi_eliptik(image_bgr: np.ndarray) -> np.ndarray:
    """Haar cascade ile yüz tespit edip her yüz için eliptik maske üretir (0/255)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, fw, fh) in faces:
        pad_x = int(0.10 * fw)
        pad_y = int(0.20 * fh)
        cx = x + fw // 2
        cy = y + fh // 2
        ax = (fw // 2) + pad_x
        ay = (fh // 2) + pad_y
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def _maskeli_bulaniklastir(image_bgr: np.ndarray, mask: np.ndarray, ksize: int = 35) -> np.ndarray:
    """Sadece mask içini bulanıklaştırır; maskeyi tüylendirip yumuşak geçiş sağlar."""
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)
    soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
    alpha = (soft.astype(np.float32) / 255.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_3c = cv2.merge([alpha, alpha, alpha])
    out = (alpha_3c * blurred + (1.0 - alpha_3c) * image_bgr).astype(np.uint8)
    return out


def yuz_bulaniklastir(gorsel_yolu: str, cilt_ile_dogrula: bool = True, kaydet: bool = False):
    # 1) Görüntüyü yükle
    img = cv2.imread(gorsel_yolu)
    if img is None:
        print("Görüntü yüklenemedi!")
        return None

    # 2) Ten rengi tespiti (HSV)
    mask_skin = _ten_maskesi_hsv(img)

    # 3) Yüz bölgesi segmentasyonu (Haar + eliptik yüz maskesi)
    mask_face = _yuz_maskesi_eliptik(img)
    if cilt_ile_dogrula:
        # Yüz maskesini ten maskesi ile kesiştirerek saç/arka plan etkisini azalt
        mask_face = cv2.bitwise_and(mask_face, mask_skin)

    # Eğer yüz bulunamazsa, orijinali döndür
    if cv2.countNonZero(mask_face) == 0:
        print("Yüz tespit edilemedi.")
        cikti = img.copy()
    else:
        # 4) Seçici bulanıklaştırma (sadece yüz alanı)
        cikti = _maskeli_bulaniklastir(img, mask_face, ksize=41)

    # Görselleştirme
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Orijinal")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask_skin, cmap='gray')
    plt.title("Ten Maskesi (HSV)")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask_face, cmap='gray')
    plt.title("Yüz Maskesi")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(cikti, cv2.COLOR_BGR2RGB))
    plt.title("Seçici Bulanıklaştırma")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    if kaydet:
        cv2.imwrite('yuz_bulanik.png', cikti)
        print('Kaydedildi: yuz_bulanik.png')

    return cikti


if __name__ == "__main__":
    yuz_bulaniklastir("yuz3.png", kaydet=True)

