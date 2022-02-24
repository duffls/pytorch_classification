from cv2 import getGaussianKernel
import numpy as np
import cv2

def getGaussianMask(ksize, sigmaX, sigmaY): 
    sigma = 0.3 * ((np.array(ksize) - 1.0) * 0.5 - 1.0) + 0.8
    if sigmaX <= 0: sigmaX = sigma[0]   # 표준편차가 양수가 아닐떄
    if sigmaY <= 0: sigmaY = sigma[1]   # 커널 사이즈로 기본 표준편차 계산
    
    u = np.array(ksize)//2
    x = np.arange(-u[0], u[0]+1, 1)     # x방향 범위
    y = np.arange(-u[1], u[1]+1, 1)     # y
    x, y = np.meshgrid(x, y)            # 좌표 행열
    
    ratio = 1/(sigmaX * sigmaY * 2 * np.pi)
    v1 = x ** 2 / (2 * sigmaX ** 2)
    v2 = y ** 2 / (2 * sigmaY ** 2)
    mask = ratio * np.exp(-(v1+v2))     # 2차원 정규분포 수식
    
    return mask / np.sum(mask)          # 원소 전체 합은 1로 유지

image = cv2.imread("book.jpg") #, cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("Error")

ksize = (17, 5)

gaussian_2d = getGaussianMask(ksize, 0, 0)
gaussian_1dx = getGaussianKernel(ksize[0], 0, cv2.CV_32F)
gaussian_1dy = getGaussianKernel(ksize[1], 0, cv2.CV_32F)

gauss_img1 = cv2.filter2D(image, -1, gaussian_2d)
gauss_img2 = cv2.GaussianBlur(image, ksize, 0)
gauss_img3 = cv2.sepFilter2D(image, -1, gaussian_1dx, gaussian_1dy)

titles = ['image', 'gauss_img1', 'gauss_img2', 'gauss_img3']

for t in titles: cv2.imshow(t, eval(t))

cv2.waitKey(0)