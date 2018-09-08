import cv2
import numpy as np


img = cv2.imread('timg_3.jpg')
h, w = img.shape[:2]  # 图像的高和宽
min_size = min(h, w)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), sigmaX=9, sigmaY=9)

X = cv2.Sobel(blur, cv2.CV_16S, 1, 0)  # X方向上的梯度计算
Y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)  # Y方向上的梯度计算
absX = cv2.convertScaleAbs(X)  # 转回uint8
absY = cv2.convertScaleAbs(Y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
canny = cv2.Canny(dst, 30, 100, 3)
circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 5, int(min_size * 0.5), param1=100, param2=1000, minRadius=int(min_size * 0.3), maxRadius=int(min_size * 0.9))

try:
    circles = np.uint16(np.around(circles))
    max_circle = circles[0, :2]
    print(max_circle)
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        # cv2.imshow('blur', canny)

        cv2.imshow('detected circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(i)
except:
    print('没有发现圆')


