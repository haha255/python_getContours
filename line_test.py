import cv2
import shapedetector
import numpy as np
import imutils
import getshellsize


def cal_dist(hist):
    dist = {}
    for gray in range(256):
        value = 0.0
        for k in range(256):
            value += hist[k][0] * abs(gray - k)
        dist[gray] = value
    return dist

if __name__ == '__main__':
    sd = getshellsize.GetShellSize()  # 实例化
    img = cv2.imread('./pic/timg_2.jpg')
    h, w = img.shape[:2]  # 图像的高和宽
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ii = cv2.pyrMeanShiftFiltering(img, 3, 100)
    grad = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    grad = sd.picGrad(grad, 0.5)
    canny = cv2.Canny(grad, 50, 150, 3)  # 这里取值不太好

    _, dst = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #dst = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 10)
    #blur = cv2.GaussianBlur(gray, (7, 7), 0)
    #gray = gray - blur
    cnts = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)  # 计算周长
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
        cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], -1, (0, 0, 255), 2)
        print(peri)
        break
    img = sd.pic_resize(img, (800, 0))
    cv2.imshow('grad', canny)
    cv2.waitKey()
    exit(0)



    res = np.uint8(np.clip((1.9 * img + 30), 0, 255))
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(res, mask, (1, 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 4)  # 横向间隔
    cv2.imshow('res', res)
    cv2.imshow('img', img)
    cv2.waitKey()
    exit(0)
    hist_array = gray_hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dst = clahe.apply(gray)
    # img = img - 70
    mv = cv2.split(img)
    sobelx = cv2.Sobel(mv[0], cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(mv[0], cv2.CV_16S, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelx = cv2.convertScaleAbs(sobelx)
    dst = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    dd = cv2.Laplacian(mv[0], cv2.CV_16S)
    dd = cv2.convertScaleAbs(dd)
    img2 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow('1', dst)
    cv2.waitKey()
    exit(0)
    # img = cv2.medianBlur(img, 9) 中值滤波
    # img = cv2.bilateralFilter(img, 25, 50, 25/2) 双边滤波，效率太低，效果也不算好
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mv = cv2.split(img)
    imglap = cv2.Laplacian(gray, cv2.CV_64F)
    img2 = cv2.adaptiveThreshold(mv[0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    img3 = cv2.medianBlur(img2, 9)
    equ = cv2.equalizeHist(gray)
    cv2.imshow('', img)
    cv2.waitKey()
    exit(0)

    img = 255 - img
    mv = cv2.split(img)
    h, w = img.shape[:2]  # 图像的高和宽
    # cv2.imshow('B', mv[0])
    # cv2.imshow('G', mv[1])
    # cv2.imshow('R', mv[2])
    # cv2.waitKey()
    # exit(0)
    blur = cv2.GaussianBlur(mv[0], (9, 9), 0)  # 高斯虚化
    # mask = sd._floodmask(blur)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 0 )
    # cv2.floodFill(img, mask, (1, 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 4)  # 横向间隔
    for i in range(5):
        cv2.floodFill(img, mask, (int(w / 5) * i + 1, 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 4)  # 横向间隔
        cv2.floodFill(img, mask, (int(w / 5) * i + 1, h - 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 4)  # 横向间隔
        cv2.floodFill(img, mask, (1, int(h / 5) * i + 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 4)  # 纵向间隔
        cv2.floodFill(img, mask, (w - 1, int(h / 5) * i + 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 4)  # 纵向间隔
    cv2.imshow('blur', img)
    cv2.waitKey()
    exit(0)
    h, w = img.shape[:2]  # 图像的高和宽
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist = cv2.equalizeHist(gray) 直方图灰度转换。效果一般
    # cv2.imshow('gray', gray)
    #cv2.imshow('hist', hist)

    # blur = cv2.GaussianBlur(gray, (9, 9), 0)  # 高斯虚化

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    # binary = sd.watershed(binary)
    cv2.imshow('binary', binary)
    cv2.waitKey()
    exit(0)
    cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序

    for c in cnts:
        peri = cv2.arcLength(c, True)  # 计算周长
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
        rect = cv2.boundingRect(c)
        #box = cv2.boxPoints(rect)
        # box = np.int0(rect)  # 最小矩形的四个脚点坐标
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),(0,255,0),2)
        #cv2.drawContours(img, [rect], -1, (0, 0, 255), 2)
        copyimg = gray[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        copyimg = 255 - copyimg
        otsu = sd.otsu(copyimg)
        _, binary = cv2.threshold(copyimg, otsu, 222, cv2.THRESH_BINARY)
        cv2.imshow('binary', binary)
        cv2.waitKey()
    exit(0)


    mask = sd._floodmask(blur)
    for i in range(10):
        cv2.floodFill(blur, mask, (int(w / 10) * i + 1, 1), (255, 255, 255), (3, 3, 3), (3, 3, 3), 8)  # 横向间隔
        cv2.floodFill(blur, mask, (int(w / 10) * i + 1, h - 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 横向间隔
        cv2.floodFill(blur, mask, (1, int(h / 10) * i + 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 纵向间隔
        cv2.floodFill(blur, mask, (w - 1, int(h / 10) * i + 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 纵向间隔


    # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    # gray = 255 - gray
    # # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # # cv2.drawContours(img, [approx], 0, (0, 255,), 2)
    # otsu = sd.otsu(gray)  # 求大津阈值
    # thresh = cv2.threshold(gray,   otsu, 255, cv2.THRESH_BINARY)[1]  # 二值化
    # kernel = np.ones((3, 3), np.uint8)  # 开运算
    # dilation = cv2.dilate(thresh, kernel, iterations=4)  # 膨胀
    # erosion = cv2.erode(dilation, kernel, iterations=4)  # 腐蚀
    #
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 取的面积最大的轮廓
    # for c in cnts:
    #     # c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # 取的面积最大的轮廓
    #     # M = cv2.moments(c)  # 计算轮廓的矩
    #     # Hu_M = cv2.HuMoments(M)  # 计算7个不变矩
    #     # print(Hu_M)
    #     # cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    #     peri = cv2.arcLength(c, True)  # 计算周长
    #     approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
    #     # 凸检查，返回是否有缺陷
    #     # print(cv2.isContourConvex(c))
    #     # 凸缺陷检测
    #     # if not cv2.isContourConvex(approx):
    #     #     hull = cv2.convexHull(approx, returnPoints=False)  #
    #     #     defects = cv2.convexityDefects(approx, hull)
    #     #     for i in range(defects.shape[0]):
    #     #         s, e, f, d = defects[i, 0]
    #     #         start = tuple(approx[s][0])
    #     #         end = tuple(approx[e][0])
    #     #         far = tuple(approx[f][0])
    #     #         cv2.circle(img, far, 5, (0, 0, 255), -1)
    #     cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
    #     rect = cv2.minAreaRect(c)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #     # cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
    # cv2.imshow('img', img)
    # cv2.imshow('erosion', erosion)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
