import cv2
import shapedetector
import numpy as np
import imutils


if __name__ == '__main__':
    sd = shapedetector.ShapeDetector()  # 实例化
    img = cv2.imread('./pic/timg_1.jpg')
    ret, approx, box = sd.detect(img)
    if not ret:
        print('没有发现矩形！')
    else:
        dst = sd.transform1(img, approx, box)
        cv2.imshow('dst', dst)
        cv2.waitKey()
        exit(0)

        # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        # cv2.drawContours(img, [approx], 0, (0, 255,), 2)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        otsu = sd.otsu(gray)  # 求大津阈值
        thresh = cv2.threshold(gray,   otsu, 255, cv2.THRESH_BINARY)[1]  # 二值化
        thresh = cv2.bitwise_not(thresh)
        kernel = np.ones((3, 3), np.uint8)  # 开运算
        dilation = cv2.dilate(thresh, kernel, iterations=4)  # 膨胀
        erosion = cv2.erode(dilation, kernel, iterations=4)  # 腐蚀
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]  # 取的面积最大的轮廓
        M = cv2.moments(c)  # 计算轮廓的矩
        Hu_M = cv2.HuMoments(M)  # 计算7个不变矩
        # print(Hu_M)
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        peri = cv2.arcLength(c, True)  # 计算周长
        approx = cv2.approxPolyDP(c, 0.005 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
        # 凸检查，返回是否有缺陷
        # print(cv2.isContourConvex(c))
        # 凸缺陷检测
        if not cv2.isContourConvex(approx):
            hull = cv2.convexHull(approx, returnPoints=False)  #
            defects = cv2.convexityDefects(approx, hull)
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                cv2.circle(dst, far, 5, (0, 0, 255), -1)
        cv2.drawContours(dst, [approx], 0, (0, 255, 0), 2)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(dst, [box], 0, (0, 0, 255), 2)
        cv2.circle(dst, (cx, cy), 5, (0, 255, 255), -1)
        cv2.imshow('img', dst)
        cv2.waitKey()
        cv2.destroyAllWindows()
