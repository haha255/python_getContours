import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import imutils
import shapedetector


fn = './PIC/timg_26.jpg'
image = cv2.imread(fn)
HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
std_shell = np.load('./std_shell.npy')


def getpos(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(HSV[y, x])


def getlinepoint(p_a, p_b, dis_c, dis_d, dis_e, dis_f):
    '''pa,pb点和长度R，返回c点和反方向d点的坐标 a位质心点'''
    if p_a[0] == p_b[0]:  # 斜率无穷，为垂直线
        p_c = (p_a[0], p_a[1] + dis_c)
        p_d = (p_a[0], p_a[1] - dis_d)
        p_e = (p_a[0] - dis_e, p_a[1])
        p_f = (p_a[0] + dis_f, p_a[1])
        return p_c, p_d, p_e, p_f
    elif p_a[1] == p_b[1]:  # 水平线
        p_c = (p_a[0] + dis_c, p_a[1])
        p_d = (p_a[0] - dis_d, p_a[1])
        p_e = (p_a[0], p_a[1] - dis_e)
        p_f = (p_a[0], p_a[1] + dis_f)
        return p_c, p_d, p_e, p_f
    else:
        radian = math.atan2(p_b[1] - p_a[1], p_a[0] - p_b[0])
        radian_90 = math.atan2(p_a[0] - p_b[0], p_a[1] - p_b[1])
        p_c = (int(p_a[0] + math.cos(radian) * dis_c), int(p_a[1] - math.sin(radian) * dis_c))
        p_d = (int(p_a[0] - math.cos(radian) * dis_d), int(p_a[1] + math.sin(radian) * dis_d))
        p_e = (int(p_a[0] + math.cos(radian_90) * dis_e), int(p_a[1] - math.sin(radian_90) * dis_e))
        p_f = (int(p_a[0] - math.cos(radian_90) * dis_f), int(p_a[1] + math.sin(radian_90) * dis_f))
        return p_c, p_d, p_e, p_f


def getcrosspoint(c, p_a, p_b, circle_c, iteration=1):
    '''根据给定的a,b两点和最小外接圆半径，求得和C形状相接的2点坐标，最后一个参数是迭代次数，一般不用超过3次。'''
    dis_s = [circle_c, circle_c, circle_c, circle_c]
    line_ret = getlinepoint(p_a, p_b, dis_s[0], dis_s[1], dis_s[2], dis_s[3])  # 返回cd两点
    for _ in range(iteration):
        dis_m = int(cv2.pointPolygonTest(c, line_ret[0], 1))  # 质点轴方向上点距离
        dis_n = int(cv2.pointPolygonTest(c, line_ret[1], 1))  # 质点轴方向下点距离
        dis_u = int(cv2.pointPolygonTest(c, line_ret[2], 1))  # 质点垂直轴方向左侧距离
        dis_v = int(cv2.pointPolygonTest(c, line_ret[3], 1))  # 质点垂直轴方向右侧距离
        dis_s[0], dis_s[1], dis_s[2], dis_s[3] = dis_s[0] + dis_m, dis_s[1] + dis_n, dis_s[2] + dis_u, dis_s[3] + dis_v
        line_ret = getlinepoint((cx, cy), (x, y), dis_s[0], dis_s[1], dis_s[2], dis_s[3])
    return line_ret


def from_points_get_rect(points):
    p_a, p_b, p_c, p_d = points  # 四点坐标
    if p_a[0] == p_b[0]:  # 垂直线
        p_ra, p_rb, p_rc, p_rd = (p_d[0], p_a[1]), (p_c[0], p_a[1]), (p_c[0], p_b[1]), (p_d[0], p_b[1])
    elif p_a[1] == p_b[1]:  # 水平线
        p_ra, p_rb, p_rc, p_rd = (p_a[0], p_d[1]), (p_a[0], p_c[1]), (p_b[0], p_c[1]), (p_b[0], p_d[1])
    else:
        k = (p_b[1] - p_a[1]) / (p_b[0] - p_a[0])  # ab线的斜率
        b1 = p_a[1] + p_a[0] / k  # 过a点平行于ab的直线截距
        b2 = p_d[1] - p_d[0] * k  # 过d点斜率为k'的直线的截距
        b3 = p_c[1] - p_c[0] * k  # 过c点
        b4 = p_b[1] + p_b[0] / k  # 过b点
        tmp_x = int((b1 - b2)/(k + 1/k))
        tmp_y = int((b2 - b1)/(k**2 + 1) + b1)
        p_ra = (tmp_x, tmp_y)
        tmp_x = int((b1 - b3) / (k + 1 / k))
        tmp_y = int((b3 - b1) / (k ** 2 + 1) + b1)
        p_rb = (tmp_x, tmp_y)
        tmp_x = int((b4 - b3) / (k + 1 / k))
        tmp_y = int((b3 - b4) / (k ** 2 + 1) + b4)
        p_rc = (tmp_x, tmp_y)
        tmp_x = int((b4 - b2) / (k + 1 / k))
        tmp_y = int((b2 - b4) / (k ** 2 + 1) + b4)
        p_rd = (tmp_x, tmp_y)
    return p_ra, p_rb, p_rc, p_rd

#th2=cv2.adaptiveThreshold(imagegray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
if __name__ == '__main__':
    # sd = shapedetector.ShapeDetector()
    # otsu = sd.otsu(HSV[:, :, 1])
    # HSV[:, :, 1] 得到图像的饱和度灰度图
    img2 = cv2.threshold(HSV[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)  # 运算核
    mb = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel, iterations=3)  # 闭运算
    cnts = cv2.findContours(mb.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)  # 计算周长
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
        if cv2.isContourConvex(approx):
            continue
        # cv2.drawContours(image, [approx], 0, (0, 255, 255), 3)
        cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
        M = cv2.moments(c)  # 计算轮廓的矩
        #Hu_M = cv2.HuMoments(M)  # 计算7个不变矩
        if M['m00'] < 1000:
            break
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        cv2.circle(image, (cx, cy), 3, (255, 0, 0), -1)  # 质心
        hull = cv2.convexHull(approx, returnPoints=False)  # 凸缺陷检测
        defects = cv2.convexityDefects(approx, hull)
        if (defects.shape[0]) != 2:
            continue
        p1 = []
        for i in range(2):
            s, e, f, d = defects[i, 0]
            # start = tuple(approx[s][0])
            # end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            p1.append(approx[f][0])
            cv2.circle(image, far, 3, (255, 0, 255), -1)  # 缺陷点标注
        y = int(p1[1][1] - (p1[1][1] - p1[0][1])/2)
        x = int(p1[1][0] - (p1[1][0] - p1[0][0])/2)
        cv2.circle(image, (x, y), 3, (255, 255, 0), -1)  # 缺陷点的中点
        #cv2.line(image, (x, y), (cx, cy), (128, 128, 0), 2)
        '''绘制最小外接圆'''
        circle = cv2.minEnclosingCircle(c)
        circle_center = (int(circle[0][0]), int(circle[0][1]))  # 圆心
        circle_r = int(circle[1])  # 半径
        # cv2.circle(image, circle_center, circle_r, (200, 0, 0), 2)
        # cv2.circle(image, circle_center, 4, (0, 0, 200), -1)
        line_2p = getcrosspoint(c, (cx, cy), (x, y), circle_r, 2)  # 返回A,B,C,D四个点
        rect_pa = from_points_get_rect(line_2p)
        # for i in range(3):
        #     cv2.line(image, rect_pa[i], rect_pa[i + 1], (0, 255, 255), 2)
        # cv2.line(image, rect_pa[0], rect_pa[3], (0, 255, 255), 2)
        # cv2.circle(image, rect_pa[0], 3, (0, 255, 255), -1)  # 外矩形的A点
        # cv2.putText(image, ('RA'), rect_pa[0], cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        # cv2.circle(image, rect_pa[1], 3, (0, 255, 255), -1)  # 外矩形的A点
        # cv2.putText(image, ('RB'), rect_pa[1], cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        # cv2.circle(image, rect_pa[2], 3, (0, 255, 255), -1)  # 外矩形的A点
        # cv2.putText(image, ('RC'), rect_pa[2], cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        # cv2.circle(image, rect_pa[3], 3, (0, 255, 255), -1)  # 外矩形的A点
        # line_p = getlinepoint((cx, cy), (x, y), circle_r, circle_r)
        # cv2.line(image, line_p[0], line_p[1], (100, 100, 50), 3)
        # dis_c = cv2.pointPolygonTest(c, line_p[0], 1)
        # dis_d = cv2.pointPolygonTest(c, line_p[1], 1)
        # line_p = getlinepoint((cx, cy), (x, y), circle_r + dis_c, circle_r + dis_d)
        # print(line_2p)
        cv2.putText(image, ('A'), line_2p[0], cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        cv2.putText(image, ('B'), line_2p[1], cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        cv2.putText(image, ('C'), line_2p[2], cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        cv2.line(image, line_2p[0], line_2p[1], (0, 255, 0), 1)
        cv2.line(image, line_2p[2], line_2p[3], (0, 255, 0), 1)
        '''写字'''
        txt = 'H:' + int(math.sqrt((line_2p[0][0] - line_2p[1][0])**2 + (line_2p[1][1] - line_2p[0][1])**2)).__str__() \
              + 'W:' + int(math.sqrt((line_2p[2][0] - line_2p[3][0])**2 + (line_2p[2][1] - line_2p[3][1])**2)).__str__()
        cv2.putText(image, txt, (cx, cy), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        rr = cv2.matchShapes(std_shell, c, cv2.CONTOURS_MATCH_I1, 0.0)
        cv2.putText(image, ('%.2f' % rr), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
            # cv2.putText()
            # print(ret)
            # ret = cv2.pointPolygonTest(c, line_p[1], 1)
            # print(ret)
            # print(circle)
            # np.column_stack()
    # print(otsu)
    # red = HSV[:, :, 1]
    # red = cv2.bitwise_not(red)
    # hist = cv2.equalizeHist(red)
    cv2.imshow('img', image)
    cv2.imwrite(fn.split('.')[0] + 'xxx.jpg', image)
    # cv2.imshow('kai', img2)
    # cv2.imshow('ret', ret)
    cv2.waitKey()

    exit(0)
    # bin = convertbin(HSV)

    print(bin)
    # cnts = cv2.findContours(bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # # 找出图像内面积最大的矩形：
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序
    # k = 0
    # for c in cnts:
    #     k += 1
    #     if k <= 8:
    #         rect = cv2.minAreaRect(c)
    #         box = cv2.boxPoints(rect)
    #         box = np.int0(box)  # 最小矩形的四个脚点坐标
    #         cv2.drawContours(image, [box], 0, (0, 0, 255), 2)


    #cv2.imshow('image', image)
    # cv2.imshow('bin', bin)
    gray_hist = cv2.calcHist([HSV[:, :, 1]], [0], None, [256], [0.0, 255.0])
    x = [i for i in range(256)]
    import matplotlib.pyplot as plt
    plt.plot(x, gray_hist)
    plt.show()
    cv2.imshow('hsv', HSV[:, :, 1])
    cv2.waitKey()
    # cv2.imshow("imageHSV", HSV)
    # cv2.imshow('image', image)
    # cv2.setMouseCallback("imageHSV", getpos)
    # cv2.waitKey()
