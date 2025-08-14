import cv2
import numpy as np
import matplotlib.pyplot as plt


def kenar_tespiti(image, canny_threshold1=50, canny_threshold2=150):
    # Boyut küçültme
    orig_height = image.shape[0]
    resized = cv2.resize(image, (800, int(800 * orig_height / image.shape[1])))

    # İşlem adımları
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Kenar iyileştirme
    kernel = np.ones((3, 3), np.uint8)
    edges_improved = cv2.dilate(edges, kernel, iterations=1)
    edges_improved = cv2.erode(edges_improved, kernel, iterations=1)

    # Kontur bulma ve belge sınırlarını tespit
    contours, _ = cv2.findContours(edges_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    document_contour = cv2.approxPolyDP(max_contour, epsilon, True)

    # Sonuç görüntüsü
    result = resized.copy()
    cv2.drawContours(result, [document_contour], -1, (0, 255, 0), 2)

    return result, edges_improved, document_contour, resized


def perspektif_duzeltme(image, contour):
    # 4 köşe noktası bulma
    if len(contour) != 4:
        contour = contour.reshape(-1, 2)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        contour = np.int32(box)
    else:
        contour = contour.reshape(-1, 2)

    # Doğru sıralama: mesafe tabanlı
    def sirala_noktalar(pts):
        # Merkezi bul
        center = pts.mean(axis=0)

        # Açılara göre sırala
        def aci_hesapla(pt):
            return np.arctan2(pt[1] - center[1], pt[0] - center[0])

        # Saat yönünün tersine sırala
        sorted_pts = sorted(pts, key=aci_hesapla)

        # Sol üst köşeden başlat
        rect = np.array(sorted_pts, dtype="float32")
        return rect

    rect = sirala_noktalar(contour)

    # Genişlik ve yükseklik hesapla
    widths = [
        np.linalg.norm(rect[1] - rect[0]),
        np.linalg.norm(rect[2] - rect[3])
    ]
    heights = [
        np.linalg.norm(rect[3] - rect[0]),
        np.linalg.norm(rect[2] - rect[1])
    ]

    max_width = int(max(widths))
    max_height = int(max(heights))

    # Standart A4 oranını koru (yükseklik > genişlik)
    if max_width > max_height:
        max_width, max_height = max_height, max_width

    # Hedef noktalar (dikey yönelim için)
    dst = np.array([
        [0, 0],  # sol üst
        [max_width, 0],  # sağ üst
        [max_width, max_height],  # sağ alt
        [0, max_height]  # sol alt
    ], dtype="float32")

    # Perspektif dönüşümü
    transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))

    return warped


def arka_plan_temizleme(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu threshold ile daha iyi arka plan ayırma
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfolojik temizleme
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned


def metin_netlestirme(image):
    # Bilateral filter ile gürültü azaltma (kenarları korur)
    filtered = cv2.bilateralFilter(image, 9, 75, 75)

    # Adaptive histogram eşitleme
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    # Unsharp masking
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return sharpened


def belge_isleme(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Görüntü yüklenemedi!")

    # 1. Kenar tespiti ve kontur bulma
    result, edges, doc_contour, resized = kenar_tespiti(image)

    # 2. Perspektif düzeltme
    warped = perspektif_duzeltme(resized, doc_contour)

    # 3. Arka plan temizleme
    cleaned = arka_plan_temizleme(warped)

    # 4. Metin netleştirme
    final = metin_netlestirme(cleaned)

    return image, result, edges, warped, cleaned, final


if __name__ == "__main__":

    original, detected, edges, warped, cleaned, final = belge_isleme("belge3.jpeg")

    # Görselleştirme
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('1. Orijinal Görüntü')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
    plt.title('2. Belge Sınırları')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('3. Kenar Tespiti')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title('4. Perspektif Düzeltme')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cleaned, cmap='gray')
    plt.title('5. Arka Plan Temizleme')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(final, cmap='gray')
    plt.title('6. Nihai Sonuç')
    plt.axis('off')

    plt.tight_layout()
    plt.show()