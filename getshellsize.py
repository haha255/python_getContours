import cv2
import numpy as np
import imutils
import math


class GetShellSize:
    def __init__(self, fn=''):
        self.fn = fn
        if self.fn != '':
            self.loadPic(self.fn)

    def loadPic(self, fn):
        self.fn = fn
        self.image = cv2.imread(self.fn)  # 直接处理，我们对原始照片不感兴趣，更希望得到A4纸张部分，因此直接处理并覆盖
        self.height, self.width = self.image.shape[:2]  # 图片的宽和高
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.floodmask = self._floodmask(self.gray)  #取洪水填充的Mask

    def picGrad(self, img, horizontal=0.5):  # horizontal取值0-1之间，计算x轴和y轴的权重
        X = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # X方向上的梯度计算
        Y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # Y方向上的梯度计算
        absX, absY = cv2.convertScaleAbs(X), cv2.convertScaleAbs(Y)  # 转回uint8
        dst = cv2.addWeighted(absX, horizontal, absY, 1 - horizontal, 0)
        return dst

    def _floodmask(self, img):  # 通过梯度找到边界，然后对边界进行二值化得到不进行填充的范围，必须给定灰度图
        grad = self.picGrad(img)
        h, w = img.shape[:2]
        canny = cv2.Canny(grad, 50, 150, 3)  # 这里取值不太好
        row = np.zeros(h, dtype=np.uint8)
        canny = np.row_stack((row, canny, row))
        col = np.zeros(w + 2, dtype=np.uint8)
        canny = np.column_stack((col, canny, col))
        return canny

    def otsu(self, img):  # 大津阈值分割算法，找阈值使用，必须给定灰度图
        gray_hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
        cv2.normalize(gray_hist, dst=gray_hist, norm_type=cv2.NORM_L1)  # 归一化
        threshold, u_t, u, p, delta = 0, 0.0, 0.0, 0.0, 0  # 初始化变量
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

    def linelength(self, p1, p2):
        return int(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    def detect_rect(self, img, morphology=4, rectarea=0.3):
        '''检测矩形，按照第三个参数，面积百分比检测，返回3个值，第一个是True/False，第二个值是近似矩形框的mat，第三个值是最小矩形框的mat or None。'''
        h, w = img.shape[:2]  # 图像的高和宽
        blur = cv2.GaussianBlur(img, (7, 7), 0)  # 高斯虚化
        ''' 洪水填充，横向取10个种子点，纵向取10个种子点，进行20次填充 '''
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        mask = self._floodmask(gray)  # mask = np.zeros([h + 2, w + 2], np.uint8)  # mask层必须必图片+2)
        for i in range(10):
            cv2.floodFill(blur, mask, (int(w / 10) * i + 1, 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 横向间隔
            cv2.floodFill(blur, mask, (int(w / 10) * i + 1, h - 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 横向间隔
            cv2.floodFill(blur, mask, (1, int(h / 10) * i + 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 纵向间隔
            cv2.floodFill(blur, mask, (w - 1, int(h / 10) * i + 1), (0, 0, 0), (5, 5, 5), (5, 5, 5), 8)  # 纵向间隔
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)  # 运算核
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 另一种核
        morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=morphology)  # 开运算，先腐蚀，后膨胀
        thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化, 自动大津值
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算周长
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
            if len(approx) == 4:  # 是4边型
                area = cv2.contourArea(c, oriented=False)  # 计算非向量面积
                if area > h * w * rectarea:
                    rect = cv2.minAreaRect(c)
                    box = np.int0(cv2.boxPoints(rect))  # 最小矩形的四个脚点坐标
                    if (cv2.contourArea(box, oriented=False)/area) >= 0.9:
                        # cv2.drawContours(img, [approx[:, 0]], -1, (255, 0, 0), 2)
                        return True, approx[:, 0], box
        return False, None, None

    def transform(self, img, paperP, rectP):  # 给定2个矩形的点，进行图形透视变换，将paperP的四点位置变换到rectP上去。图像矫正
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
        if rP_d[0] - rP_a[0] >= rP_a[1] - rP_d[1]:  # 夹角为锐角的情形
            final_rect = np.array([[0, dis_ab], [0, 0], [dis_ad, 0], [dis_ad, dis_ab]],
                                  dtype=np.float32)  # final_rect = np.float32(final_rect)
            dis_ab, dis_ad = dis_ad, dis_ab
        else:
            final_rect = np.array([[dis_ab, dis_ad], [0, dis_ad], [0, 0], [dis_ab, 0]],
                                  dtype=np.float32)  # final_rect = np.float32(final_rect)
        M = cv2.getPerspectiveTransform(paperP, final_rect)  # 生成透视变换矩阵
        dst = cv2.warpPerspective(img, M, (dis_ab, dis_ad))  # 透视变换
        return dst

    def detect_shell(self, img, maxnum=8, sim=0.9):
        std_shell = np.load('./std_shell.npy')  # 标准扇贝轮廓
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转为HSV图像格式 HSV[:, :, 1] 得到图像的饱和度灰度图
        thres = cv2.threshold(HSV[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # 二值化，大津自动阈值
        kernel = np.ones((3, 3), np.uint8)  # 运算核
        mb = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel, iterations=3)  # 闭运算
        cnts = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序
        count = 0
        for c in cnts:
            similar = cv2.matchShapes(std_shell, c, cv2.CONTOURS_MATCH_I1, 0.0)
            if similar <= sim:  # 相似度不高
                continue
            peri = cv2.arcLength(c, True)  # 计算周长
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 用多边形拟合形状，第二个参数取值一般是1-5%的周长
            if cv2.isContourConvex(approx):  # 凸多边形就跳过
                continue
            hull = cv2.convexHull(approx, returnPoints=False)  # 凸缺陷检测
            defects = cv2.convexityDefects(approx, hull)
            if (defects.shape[0]) != 2:  # 缺陷不是2就跳过，扇贝有2个腰点，通过凸检测能测得到。
                continue
            p1 = []
            for i in range(2):
                s, e, f, d = defects[i, 0]
                # start = tuple(approx[s][0])
                # end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                p1.append(approx[f][0])
                cv2.circle(image, far, 3, (255, 0, 255), -1)  # 缺陷点标注
            y = int(p1[1][1] - (p1[1][1] - p1[0][1]) / 2)
            x = int(p1[1][0] - (p1[1][0] - p1[0][0]) / 2)
            cv2.circle(image, (x, y), 3, (255, 255, 0), -1)  # 缺陷点的中点



            cv2.drawContours(img, [c], -1, (0, 0, 255), 2)  # 绘制轮廓线
            M = cv2.moments(c)  # 计算轮廓的矩
            # Hu_M = cv2.HuMoments(M)  # 计算7个不变矩
            if M['m00'] < 1000:
                break
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            cv2.circle(image, (cx, cy), 3, (255, 0, 0), -1)  # 质心



            # cv2.line(image, (x, y), (cx, cy), (128, 128, 0), 2)
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
            txt = 'H:' + int(
                math.sqrt((line_2p[0][0] - line_2p[1][0]) ** 2 + (line_2p[1][1] - line_2p[0][1]) ** 2)).__str__() \
                  + 'W:' + int(
                math.sqrt((line_2p[2][0] - line_2p[3][0]) ** 2 + (line_2p[2][1] - line_2p[3][1]) ** 2)).__str__()
            cv2.putText(image, txt, (cx, cy), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
            rr = cv2.matchShapes(std_shell, c, cv2.CONTOURS_MATCH_I1, 0.0)
            cv2.putText(image, ('%.2f' % rr), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))

    def test(self):
        print(self.width, self.height)

if __name__ == '__main__':
    shells = GetShellSize('s1.jpg')
    shells.test()
