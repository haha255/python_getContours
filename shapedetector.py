import cv2
import numpy as np
import imutils
import math


class ShapeDetector:
    def __init__(self):
        pass

    def otsu(self, img):
        '''大津阈值分割算法，找阈值使用'''
        gray_hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
        cv2.normalize(gray_hist, dst=gray_hist, norm_type=cv2.NORM_L1)  # 归一化
        threshold, u_t, u, p, delta = 0, 0.0, 0.0, 0.0, 0
        u_t = sum([index * value for index, value in enumerate(gray_hist)])  # 整个图像的平均灰度
        for index, value in enumerate(gray_hist):
            u += index * value
            p += value
            t = u_t * p - u
            if p == 0 or p == 1:
                delta_tmp = 0
            else:
                delta_tmp = t * t/(p * (1 - p))
            if delta_tmp > delta:
                delta = delta_tmp
                threshold = index
        return threshold

    def detect(self, img, morphology=4, rectarea=0.3):
        '''检测矩形，按照第三个参数，面积百分比检测，返回3个值，第一个是True/False，第二个值是近似矩形框的mat，第三个值是最小矩形框的mat or None。'''
        h, w = img.shape[:2]  # 图像的高和宽
        blur = cv2.GaussianBlur(img, (7, 7), 0)  # 高斯虚化
        ''' 洪水填充，横向取10个种子点，纵向取10个种子点，进行20次填充 '''
        mask = self._floodmask(blur)  # mask = np.zeros([h + 2, w + 2], np.uint8)  # mask层必须必图片+2)
        for i in range(10):
            cv2.floodFill(blur, mask, (int(w/10) * i + 1, 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 横向间隔
            cv2.floodFill(blur, mask, (int(w / 10) * i + 1, h - 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 横向间隔
            cv2.floodFill(blur, mask, (1, int(h/10) * i + 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 纵向间隔
            cv2.floodFill(blur, mask, (w - 1, int(h / 10) * i + 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 纵向间隔
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)  # 运算核
        # cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 另一种核
        dilation = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=morphology)  # 开运算，先腐蚀，后膨胀
        thresh = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化, 自动大津值
        # thresh = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化, 自动大津值
        cv2.imshow('i', thresh)

        # cv2.imshow('li', thresh1)
        cv2.waitKey()
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        '''找出图像内面积最大的矩形：'''
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算周长
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
            if len(approx) == 4:
                area = cv2.contourArea(c, oriented=False)  # 计算非向量面积
                if area > h * w * rectarea:
                    rect = cv2.minAreaRect(c)
                    box = np.int0(cv2.boxPoints(rect))  # 最小矩形的四个脚点坐标
                    if (cv2.contourArea(box, oriented=False)/area) >= 0.9:
                        cv2.drawContours(img, [approx[:, 0]], -1, (255, 0, 0), 2)
                        cv2.imshow('uii', img)
                        cv2.waitKey()
                        return True, approx[:, 0], box
        return False, None, None

    def _floodmask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X = cv2.Sobel(gray, cv2.CV_16S, 1, 0)  # X方向上的梯度计算
        Y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)  # Y方向上的梯度计算
        absX = cv2.convertScaleAbs(X)  # 转回uint8
        absY = cv2.convertScaleAbs(Y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        canny = cv2.Canny(dst, 50, 150, 3)
        ww, hh = canny.shape
        row = np.zeros(hh, dtype=np.uint8)
        canny = np.row_stack((row, canny, row))
        col = np.zeros(ww + 2, dtype=np.uint8)
        canny = np.column_stack((col, canny, col))
        return canny

    def transform(self, img, paperP, rectP):
        '''给定2个矩形的点，进行图形透视变换，将paperP的四点位置变换到rectP上去。'''
        rP_a, rP_b, rP_c, _ = rectP  # 标准矩形的角点
        # pp_a, pp_b, pp_c, pp_d = paperP  # 识别的纸张角点
        w_rect = self.linelength(rP_a, rP_b)  # a-b的长度
        h_rect = self.linelength(rP_c, rP_b)  # b-c的长度，也就是a-d的长度
        tmp = []
        for i in range(4):
            tmp.append(self.linelength(rP_a, paperP[i]))
        pp_a = paperP[tmp.index(min(tmp))]
        pp_c = paperP[tmp.index(max(tmp))]
        tmp = []
        for i in range(4):
            tmp.append(self.linelength(rP_b, paperP[i]))
        pp_b = paperP[tmp.index(min(tmp))]
        pp_d = paperP[tmp.index(max(tmp))]
        paperP = np.float32([pp_a, pp_b, pp_c, pp_d])
        print(paperP)
        # final_rect = np.array([[w_rect, h_rect], [0, h_rect], [0, 0], [w_rect, 0]], dtype=np.float32)
        final_rect = np.array([[w_rect, h_rect], [0, h_rect], [0, 0], [w_rect, 0]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(paperP, final_rect)  # 生成透视变换矩阵
        dst = cv2.warpPerspective(img, M, (w_rect, h_rect))
        return dst
    def transform1(self, img, paperP, rectP):  # 给定2个矩形的点，进行图形透视变换，将paperP的四点位置变换到rectP上去。图像矫正
        rP_a, rP_b, _, rP_d = rectP  # 标准矩形的角点
        dic_rec = []  # 记录纸张和标准矩形角角相对
        dis_ab = self.linelength(rP_a, rP_b)  # a-b的长度
        dis_ad = self.linelength(rP_a, rP_d)  # a-d的长度
        for i in range(4):  # 矩形，直接写死4个点
            tmp, ii = max(dis_ab, dis_ad), 0
            for index, value in enumerate(paperP):
                length = self.linelength(rectP[i], value)
                if length < tmp:
                    tmp, ii = length, index
            dic_rec.append(paperP[ii])
        paperP = np.float32(dic_rec)
        if rP_d[0] - rP_a[0] >= rP_a[1] - rP_d[1]:  # 夹角为45角的情形
            final_rect = np.array([[0, dis_ab], [0, 0], [dis_ad, 0], [dis_ad, dis_ab]],
                                  dtype=np.float32)  # final_rect = np.float32(final_rect)
            dis_ab, dis_ad = dis_ad, dis_ab  # 交换长宽数据
        else:
            final_rect = np.array([[dis_ab, dis_ad], [0, dis_ad], [0, 0], [dis_ab, 0]],
                                  dtype=np.float32)  # final_rect = np.float32(final_rect)
        # final_rect = np.array([[dis_ab, dis_ad], [0, dis_ad], [0, 0], [dis_ab, 0]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(paperP, final_rect)  # 生成透视变换矩阵
        dst = cv2.warpPerspective(img, M, (dis_ab, dis_ad))  # 透视变换
        return dst

    def linelength(self, p1, p2):
        return int(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    def watershed(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow('', binary)
        cv2.waitKey()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算，先腐蚀，再膨胀
        sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 膨胀运算
        dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)  # 距离转换，转成灰度图，具体还没有理解
        dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
        _, surface = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)
        surface_fg = np.uint8(surface)
        unknown = cv2.subtract(sure_bg, surface_fg)
        ref, markers = cv2.connectedComponents(sure_bg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers=markers)
        img[markers == -1] = [0, 0, 255]
        return img


