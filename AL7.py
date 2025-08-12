import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def hough_lines(kenar_goruntusu, rho_resolution=1, theta_resolution=np.pi / 180):
    """
    Hough transform ile çizgi tespiti

    Parametreler:
    kenar_goruntusu -- Binary kenar görüntüsü (2D numpy array)
    rho_resolution -- Rho değerlerinin çözünürlüğü (default: 1 piksel)
    theta_resolution -- Theta değerlerinin çözünürlüğü (default: 1 derece)

    Returns:
    accumulator -- Hough uzayındaki oy matrisi
    rhos -- Rho değerleri array'i
    thetas -- Theta değerleri array'i
    """

    # Görüntü boyutlarını al
    height, width = kenar_goruntusu.shape

    # Maksimum rho değerini hesapla (görüntü köşegeni)
    max_rho = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    rhos = np.arange(-max_rho, max_rho + 1, rho_resolution)

    # Theta değerlerini oluştur (0 ile pi arasında)
    thetas = np.arange(0, np.pi, theta_resolution)

    # Accumulator matrisini oluştur (rhos x thetas)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Kenar piksellerinin koordinatlarını bul
    y_idxs, x_idxs = np.nonzero(kenar_goruntusu)

    # Her kenar pikseli için
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        # Her theta değeri için
        for theta_idx in range(len(thetas)):
            theta = thetas[theta_idx]

            # Rho değerini hesapla: rho = x*cos(theta) + y*sin(theta)
            rho = x * np.cos(theta) + y * np.sin(theta)

            # Rho'yu en yakın rho değerine yuvarla ve indeksini bul
            rho_idx = np.argmin(np.abs(rhos - rho))

            # Accumulator'u güncelle
            accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas

# Görüntüyü yükle ve kenarları tespit et
image = cv2.imread("C:/Users/sudea/Downloads/lena (1).png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 50, 150)

# Hough Transform uygula
accumulator, rhos, thetas = hough_lines(edges)

# Sonuçları görselleştir
plt.imshow(accumulator, cmap='hot',
           extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]),
                   rhos[-1], rhos[0]])
plt.xlabel('Theta (degrees)')
plt.ylabel('Rho (pixels)')
plt.title('Hough Accumulator')
plt.colorbar()
plt.show()