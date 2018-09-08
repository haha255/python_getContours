import cv2
import numpy as np
import imutils
import shapedetector


img = cv2.imread('timg_4.jpg')
#  img = cv2.resize(img, 1080)
h, w = img.shape[:2]  # 图像的高和宽
blur = cv2.GaussianBlur(img, (7, 7), 0)
#  洪水填充，以左上角为种子点，填充
mask = np.zeros([h + 2, w + 2], np.uint8)  # mask层必须必图片+2)
cv2.floodFill(blur, mask, (w - 1, h - 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 右下角
cv2.floodFill(blur, mask, (1, 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 左上角
cv2.floodFill(blur, mask, (w - 1, 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 左下角
cv2.floodFill(blur, mask, (1, h - 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 右上角
# cv2.imshow('olddilation', blur)
# cv2.waitKey(0)
min_size = min(h, w)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(blur, kernel, iterations=4)  # 腐蚀一次
dilation = cv2.dilate(erosion, kernel, iterations=4)  # 膨胀一次


(_, thresh) = cv2.threshold(dilation, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow('dilation', dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
for c in cnts:
    peri = cv2.arcLength(c, True)  # 计算周长
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)  #用多边形拟合形状，第二个参数取值一般是1-5%的周长
    if len(approx) == 4:
        area = cv2.contourArea(c, oriented=False)
        if area > h * w * 0.3:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            pts1 = np.float32(box)
            pts2 = np.float32(approx)
            x, y = rect[0]  # 中心坐标
            width, height = rect[1]  # 长宽，width >=height
            angle = rect[2]  # 角度-90 至 0
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            cv2.drawContours(img, [approx], 0, (0, 255, ), 2)
            aa = cv2.contourArea(box, oriented=False)
            print(aa, area, area/aa)
            ll = shapedetector.ShapeDetector()
            ret, a, b = ll.detect(img)
            if ret:
                print(a)
                print('---')
                print(b)
            #print(approx[:, 0])
            dst = ll.transform(img, approx[:, 0], box)

cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
X = cv2.Sobel(blur, cv2.CV_16S, 1, 0)  # X方向上的梯度计算
Y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)  # Y方向上的梯度计算
absX = cv2.convertScaleAbs(X)  # 转回uint8
absY = cv2.convertScaleAbs(Y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(dst, kernel, iterations=3)  # 腐蚀一次
dilation = cv2.dilate(erosion, kernel, iterations=3)  # 膨胀一次
# opening = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
canny = cv2.Canny(dilation, 50, 150, 3)
lines = cv2.HoughLinesP(canny, rho=1.0, theta=np.pi/180, threshold=100, minLineLength=min_size * 0.1, maxLineGap=min_size * 0.3)
lines1 = lines[:,0,:]#提取为二维
print(len(lines1))
for x1, y1, x2, y2 in lines1[:]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

cv2.imshow('thresh', thresh)
cv2.imshow('dst', dst)
cv2.imshow('canny', canny)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''