import numpy as np
import cv2
from Commons.filters import filter

def differential(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data2, np.float32).reshape(3, 3)
    
    dst1 = filter(image, mask1)
    dst2 = filter(image, mask2)
    dst1, dst2 = np.abs(dst1), np.abs(dst2)
    
    dst = cv2.magnitude(dst1, dst2)
    
    dst = np.clip(dst, 0, 255).astype('uint8')
    dst = np.clip(dst1, 0, 255).astype('uint8')
    dst = np.clip(dst2, 0, 255).astype('uint8')
    
    return dst, dst1, dst2
    
image = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception

data1 = [-1,0,0,
         0,1,0,
         0,0,0]
data2 = [0,0,-1,
         0,1,0,
         0,0,0]

dst, dst1, dst2 = differential(image, data1, data2)

cv2.imshow('image', image)
cv2.imshow('roberts edge', dst)
cv2.imshow('dist1', dst1)
cv2.imshow('dist2', dst2)
cv2.waitKey(0)
