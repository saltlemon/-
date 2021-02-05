import cv2
import matplotlib.pyplot as plt
import numpy as np
def showing(jpg_img,alpha):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(jpg_img)
    plt.title("img")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(alpha)
    plt.title("Matting Alpha img")
    plt.axis("off")
    plt.show()

jpg = cv2.imread('7.jpg', cv2.IMREAD_UNCHANGED)
jpg = cv2.resize(jpg, (400,256))
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(jpg)
plt.title("img")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(jpg)
plt.title("Matting Alpha img")
plt.axis("off")
plt.show()