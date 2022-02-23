#캐니 에지 검출
"""
1) Gaussian bulrring을 통해 노이즈 제거
2) 소벨 마스크를 이용하여 기울기 강도와 방향 검출
3) non-maximum suppression
4) 이력 임계값(hythesis threshold)로 에지 검출
"""
import numpy as np, cv2
def nonmax_suprression(sobel, direct):          #3)에 사용되는 non-maximum suprression 함수 구현
    rows, cols = sobel.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    for i in range(1, rows -1):
        for j in range(1, cols -1):                         #관심 영역을 통해 이웃 화소 가져오기
            values = sobel[i-1:i+2, j-1:j+2].flatten()      # 중심 에지 주변의 9개의 화소를 가져오기
            first = [3, 0, 1, 2]                            # 첫 이웃 화소 좌표 4개
            id = first[direct[i, j]]                        # 방향에 따른 첫 이웃화소 위치
            v1, v2 = values[id], values[8-id]
            dst[i,j] = sobel[i,j] if (v1 < sobel[i, j] > v2) else 0
    return dst
def trace(max_sobel, i, j, low):
    h, w = max_sobel.shape
    if (0<=i <h and 0 <=j < w) == False: return
    if pos_ck[i,j] > 0 and max_sobel[i, j] > low:
        pos_ck[i, j] = 255
        canny[i ,j] = 255
        trace(max_sobel, i-1, j-1, low)
        trace(max_sobel, i, j-1, low)
        trace(max_sobel, i+1, j-1, low)
        trace(max_sobel, i-1, j, low)
        trace(max_sobel, i+1, j, low)
        trace(max_sobel, i-1, j+1, low)
        trace(max_sobel, i, j+1, low)
        trace(max_sobel, i+1, j+1, low)
def hythesis_th(max_sobel, low, high):
    rows, cols = max_sobel.shape[:2]
    for i in range(1, rows- 1):
        for j in range(1, cols-1):
            if max_sobel[i,j] >= high: trace(max_sobel, i, j, low)
image = cv2.imread("./image/test.jpg", cv2.IMREAD_GRAYSCALE)
pos_ck = np.zeros(image.shape[:2], np.uint8)
canny = np.zeros(image.shape[:2], np.uint8)
gaus_img = cv2.GaussianBlur(image, (5,5), 0.3)
Gx = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F,1,0,3)
Gy = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F,0,1,3)
sobel = cv2.magnitude(Gx,Gy)
directs = cv2.phase(Gx,Gy) / (np.pi/4)
directs = (directs.astype(int)%4)
max_sobel=nonmax_suprression(sobel, directs)
hythesis_th(max_sobel, 100, 150)
canny2 = cv2.Canny(image, 100, 150)
cv2.imshow("image", image)
#cv2.imshow("canny", canny)
cv2.imshow("OpenCV_Canny", canny2)
cv2.waitKey(0)