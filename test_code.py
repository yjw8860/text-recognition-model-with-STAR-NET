import cv2
import numpy as np
import matplotlib.pyplot as plt

num_cluster = 2
image = cv2.imread('D:\data\OCR_DATASET\imgs/book_cover_art_000033_1.jpg', cv2.IMREAD_GRAYSCALE)
pixels = image.flatten()
median = int(np.median(pixels))
binary01 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('./test.jpg', binary01)