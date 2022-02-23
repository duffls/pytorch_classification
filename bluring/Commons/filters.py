import numpy as np
import cv2

# convolution 수행하는 함수 구현
def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32) # 회선 결과를 저장하는 행렬
    #ycenter, xcenter = rows//2, cols//2
    ycenter, xcenter = mask.shape[0]//2, mask.shape[1]//2
    
    for i in range(ycenter, rows - ycenter):
        for j in range(xcenter, cols - xcenter): # xcenter, cols - xcenter):
            y1, y2 = i-ycenter, i+ycenter+1             # ROI영역 높이 범위
            x1, x2 = j-xcenter, j+xcenter+1             # ROI영역 너비 범위
            roi = image[y1:y2, x1:x2].astype('float32') # ROI영역 형변환
            tmp = cv2.multiply(roi, mask)               # 회선 연산 적용, 원소간 곱
            dst[i, j] = cv2.sumElems(tmp[0])            # 출력 화소 저장
    # for i in range(0, rows - mask.shape[0]):
    #     for j in range(0, cols - mask.shape[1]):
    #         y1, y2 = i, i+mask.shape[0]
    #         x1, x2 = j, j+mask.shape[1]
    #         roi = image[y1:y2, x1:x2].astype('float32')
    #         tmp = cv2.multiply(roi, mask)               # 회선 연산 적용, 원소간 곱
    #         dst[i, j] = cv2.sumElems(tmp[0])            # 출력 화소 저장
            
    return dst

# 픽셀에 접근해서 연산
def filter2(image, mask): 
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    ycenter, xcenter = rows//2, cols//2
    
    for i in range(rows-mask.shape[0]): # ycenter):
        for j in range(cols-mask.shape[1]): # xcenter):
            sum = 0.0
            for u in range(mask.shape[0]):          # 마스크 원소 순회
                for v in range(mask.shape[1]):
                    y, x = i + u, j + v
                    sum += image[y, x] * mask[u, v] # 회선 결과 저장
            dst[i, j] = sum
    # for i in range(ycenter):
    #     for j in range(xcenter):
    #         sum = 0.0
    #         for u in range(mask.shape[0]):          # 마스크 원소 순회
    #             for v in range(mask.shape[1]):
    #                 #y, x = i + u - ycenter, j + v - xcenter
    #                 sum += image[y, x] * mask[u, v] # 회선 결과 저장
    #         dst[i, j] = sum
    return dst

def differential(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data2, np.float32).reshape(3, 3)
    
    dst1 = filter(image, mask1)
    dst2 = filter(image, mask2)
    dst = cv2.magnitude(dst1, dst2)
    
    dst = cv2.convertScaleAbs(dst)
    dst1 = cv2.convertScaleAbs(dst1)
    dst2 = cv2.convertScaleAbs(dst2)
    
    return dst, dst1, dst2

