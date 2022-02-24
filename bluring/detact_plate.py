import numpy as np
import cv2

image = cv2.imread('car_number_2.jpg', cv2.IMREAD_COLOR)

mask = np.ones((5, 17), np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (5, 5))
gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 5)

th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, mask, iterations=3)

cv2.imshow("image", image)
cv2.imshow("binary image", th_img)
cv2.imshow("opening", morph)
cv2.waitKey()