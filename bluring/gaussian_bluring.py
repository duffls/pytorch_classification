import numpy as np
import cv2

image = cv2.imread("book.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception('Error')

gaus = cv2.GaussianBlur(image, (7, 7), 0, 0)
dist1 = cv2.Laplacian(gaus, cv2.CV_16S, 7)

gaus1 = cv2.GaussianBlur(image, (3, 3), 0)
gaus2 = cv2.GaussianBlur(image, (9, 9), 0)
dist2 = gaus1 - gaus2           # 미분계수가 다른 필터의 차이를 통해 마스크 생성

cv2.imshow("image", image)
cv2.imshow("dist1 - Laplacian of Gaussian", dist1.astype("uint8"))
cv2.imshow("dist2 - Difference of GAussian", dist2)
cv2.waitKey(0)
