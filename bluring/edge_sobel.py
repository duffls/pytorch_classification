import numpy as np
import cv2
from Commons.filters import differential

image = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception

data1 = [-1,0,1,
         -2,0,2,
         -1,0,1]
data2 = [-1,-2,-1,
         0,0,0,
         1,2,1]

dst, dst1, dst2 = differential(image, data1, data2)

dst3 = cv2.Sobel(np.float32(image), cv2.CV_32F, 1, 0, 3)
dst4 = cv2.Sobel(np.float32(image), cv2.CV_32F, 0, 1, 3)

dst3 = cv2.convertScaleAbs(dst3)
dst4 = cv2.convertScaleAbs(dst4)

cv2.imshow('image', image)
cv2.imshow('dist1 - vertical_mask', dst1)
cv2.imshow('dist2 - horizental_mask', dst2)
cv2.imshow('dist3 - vertical_OpenCV', dst3)
cv2.imshow('dist4 - horizental_OpenCV', dst4)
cv2.waitKey(0)
