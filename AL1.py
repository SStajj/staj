import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

arti =np.zeros((10,10), dtype=np.uint8)
arti [5,:] = 255
arti [:,5] = 255

plt.imshow(arti, 'gray')
plt.colorbar()
plt.title('10x10 beyaz arti')
plt.show()

print(f"Goruntu boyutu: {arti.shape}")
