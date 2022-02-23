# filter_average

import numpy as np
import cv2

def average_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize//2
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = center, i+center+1
            x1, x2 = center, j+center+1
            if y1 < 0 or y2 > rows or x1 < 0 or x2 > cols:
                dst[i, j] = image[i, j]
            else:
                mask = image[y1:y2, x1:x2]
                dst[i, j] = cv2.mean(mask)[0]
    return dst

image = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception('Error')

avg_img = average_filter(image, 5)
blur_img = cv2.blur(image, (5, 5), cv2.BORDER_REFLECT)
box_img = cv2.boxFilter(image, ddepth=-1, ksize=(5, 5))

cv2.imshow('image', image)
cv2.imshow('avg_img', avg_img)
cv2.imshow('blur_img', blur_img)
cv2.imshow('box_img', box_img)

cv2.waitKey(0)