import numpy as np
import cv2
from Commons.filters import differential

image = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception

data1 = [-1,0,1,
         -1,0,1,
         -1,0,1]
data2 = [-1,-1,-1,
         0,0,0,
         1,1,1]

dst, dst1, dst2 = differential(image, data1, data2)

cv2.imshow('image', image)
cv2.imshow('prewitt edge', dst)
cv2.imshow('dist1', dst1)
cv2.imshow('dist2', dst2)
cv2.waitKey(0)
