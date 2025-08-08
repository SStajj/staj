import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

goruntu = np.array([
    [0, 50, 100, 150, 200],
    [50, 100, 150, 200, 250],
    [100, 150, 200, 250, 255],
    [150, 200, 250, 255, 200],
    [200, 250, 255, 200, 150]
])

plt.imshow(goruntu, cmap ='gray') #goruntuyu goster gri tonlariyla
plt.colorbar() #sagda renk deger cubugu(kac ne demek)
plt.title('5x5 Piksel goruntu')#baslikasasdad
plt.show() #goster

print(f"Goruntu boyutu: {goruntu.shape}")
print(f"(2,3) konumundaki piksel degeri: {goruntu[2,3]}")