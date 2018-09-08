import cv2
import numpy as np
import shapedetector
import matplotlib.pyplot as plt


def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    hist = cv2.normalize(hist, 1.0)
    # print(hist)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)
    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    return histImg

ll = shapedetector.ShapeDetector()
img = cv2.imread('s4.jpg')
h, w = img.shape[:2]  # 图像的高和宽
blured = cv2.blur(img, (4, 4))  # 去掉噪声
copyImg = blured.copy()
mask = np.zeros([h + 2, w + 2], np.uint8)  # mask层必须必图片+2
gray = cv2.cvtColor(copyImg, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])
hist = cv2.normalize(hist, 1.0)
x = np.array([i for i in range(256)])


otsu = ll.otsu(blur)
print(otsu)

_, th = cv2.threshold(blur, otsu, 255, cv2.THRESH_BINARY)
cv2.imshow('', th)
cv2.waitKey()

# canny = cv2.Canny(gray, otsu, threshold1=)
plt.plot(x, hist)
plt.show()
exit(0)
hist = calcAndDrawHist(gray, [0, 0, 255])
cv2.imshow('', hist)
cv2.waitKey()


# cv2.imshow(hist)
# cv2.waitKey()
#这里执行漫水填充，参数代表：
#copyImg：要填充的图片
#mask：遮罩层
#(30,30)：开始填充的位置（开始的种子点）
#(0,255,255)：填充的值，这里填充成黄色
#(100,100,100)：开始的种子点与整个图像的像素值的最大的负差值
#(50,50,50)：开始的种子点与整个图像的像素值的最大的正差值
#cv.FLOODFILL_FIXED_RANGE：处理图像的方法，一般处理彩色图象用这个方法
cv2.floodFill(copyImg, mask, (w - 1, h - 1), (0, 0, 0), (3, 3, 3), (2, 2, 2), 8)
#gray = cv2.cvtColor(copyImg, cv2.COLOR_BGR2GRAY)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
#opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
#closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

cv2.imshow('floodfill', copyImg)
cv2.waitKey(0)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.GaussianBlur(gray, (5, 5), 0)
binary = cv2.Canny(binary, 35, 125)

ret, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)  # 第一个参数为输入图像，第二个为阈值，第三个输出图像的最大值，第四个为阈值类型
# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 15)  # 算数平均法的自适应二值化,第一个参数输入图像，2：输出图像最大值，3：算法，4：阈值类型，5：blocksize，6：常数
# binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
# cv2.imshow('img', binary)
# cv2.waitKey(0)
'''
cv2.imshow('img', binary)
cv2.waitKey(0)
'''
binary, contours, opt = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 返回3个值，第一个是所处理的图像，第二个是才是轮廓，第三个各层轮廓的索引
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)  # 第一个参数为图片，第二个参数为轮廓数组，第三个为轮廓索引，-1为全部轮廓，后面的是轮廓颜色和线宽
cv2.imshow('img', img)
cv2.waitKey(0)
