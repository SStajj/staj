import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_lines(kenar_goruntusu, rho_resolution=1, theta_resolution=np.pi / 180):
    h, w = kenar_goruntusu.shape
    max_rho = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    rhos = np.arange(-max_rho, max_rho + 1, rho_resolution)
    thetas = np.arange(0, np.pi, theta_resolution)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(kenar_goruntusu)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx, theta in enumerate(thetas):
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            rho_idx = np.where(rhos == rho)[0][0]
            accumulator[rho_idx, t_idx] += 1
    return accumulator, rhos, thetas


# === Uygulama ===
# 1. Görüntüyü yükle ve griye çevir
img = cv2.imread("lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Kenar tespiti (Canny)
edges = cv2.Canny(gray, 100, 200)

# 3. Hough transformunu uygula
accumulator, rhos, thetas = hough_lines(edges)

# 4. En güçlü çizgileri bul (ör: en yüksek 10 değer)
num_lines = 10
indices = np.argpartition(accumulator.flatten(), -num_lines)[-num_lines:]
rho_theta_pairs = [np.unravel_index(i, accumulator.shape) for i in indices]

# 5. Çizgileri orijinal görüntüye çiz
output_img = img.copy()
for rho_idx, theta_idx in rho_theta_pairs:
    rho = rhos[rho_idx]
    theta = thetas[theta_idx]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 6. Sonuçları göster
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Orijinal")
plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title("Kenarlar")
plt.subplot(133), plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)), plt.title("Çizgiler")
plt.show()
