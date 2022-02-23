import numpy as np
import cv2
from Commons.filters import filter, filter2

image = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

data = [1/9, 1/9, 1/9,
        1/9, 1/9, 1/9,
        1/9, 1/9, 1/9]  # 블러링 마스크 원소

mask = np.array(data, np.float32).reshape(3, 3) # 마스크 행렬 생성

blur1 = filter(image, mask)     #blur1: 행렬처리방식 회선 수행
blur2 = filter2(image, mask)    #blur2: 화소접근방식 회선 수행

blur1 = blur1.astype('uint8')   # 행렬표시를 위한 형변환(윈도우에 영상으로 표시하기위한),
blur2 = cv2.convertScaleAbs(blur2)

cv2.imshow("image", image)
#cv2.imshow("blur1", blur1)
cv2.imshow("blur2", blur2)

cv2.waitKey(0)
