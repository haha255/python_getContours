import cv2
import numpy as np
import imutils
import math
import os


class GetShellSize:
    def __init__(self, fn=''):
        self.fn = fn
        self.papersize = (297, 210)  # 纸张大小，毫米单位，宽边（大值）在前。
        self.rate = 0  # 像素与纸张毫米的比率 毫米数/像素数
        self.img = []
        if self.fn != '':
            img = self.loadPic(self.fn)

    def loadPic(self, fn):
        self.fn = fn
        self.img = cv2.imread(self.fn)
        return self.img
        '''self.image = cv2.imread(self.fn)  # 直接处理，我们对原始照片不感兴趣，更希望得到A4纸张部分，因此直接处理并覆盖
        self.height, self.width = self.image.shape[:2]  # 图片的宽和高
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.floodmask = self._floodmask(self.gray)  #取洪水填充的Mask'''

    def setpapersize(self, papersize=(297, 210)):
        '''设置纸张大小，尺寸大的数字放在第一个。'''
        self.papersize = papersize

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
        row = np.zeros(w, dtype=np.uint8)
        canny = np.row_stack((row, canny, row))
        col = np.zeros(h + 2, dtype=np.uint8)
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
        # blur = cv2.GaussianBlur(img, (7, 7), 0)  # 高斯虚化
        blur = cv2.pyrMeanShiftFiltering(img, 3, 50)  #
        ''' 洪水填充，横向取10个种子点，纵向取10个种子点，进行20次填充 '''
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        mask = self._floodmask(gray)  # mask = np.zeros([h + 2, w + 2], np.uint8)  # mask层必须必图片+2)
        for i in range(10):
            cv2.floodFill(blur, mask, (int(w / 10) * i + 1, 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 横向间隔
            cv2.floodFill(blur, mask, (int(w / 10) * i + 1, h - 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 横向间隔
            cv2.floodFill(blur, mask, (1, int(h / 10) * i + 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 纵向间隔
            cv2.floodFill(blur, mask, (w - 1, int(h / 10) * i + 1), (0, 0, 0), (3, 3, 3), (3, 3, 3), 8)  # 纵向间隔
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)  # 运算核
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 另一种核
        morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=morphology)  # 开运算，先腐蚀，后膨胀
        thresh = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 10)
        # thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化, 自动大津值
        thresh = thresh + morph
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

    def detect_shell(self, img, maxnum=10, sim=0.8):
        h, w = img.shape[:2]
        tmpmax, tmpmin = self.papersize
        if tmpmax < tmpmin:
            tmpmax, tmpmin = tmpmin, tmpmax
        if h > w:
            self.rate = tmpmax / h  # 按照最大的边计算像素尺寸比率
            w = int(tmpmin / self.rate)
        else:
            self.rate = tmpmax / w
            h = int(tmpmin / self.rate)  # 计算按照相同的比率，高度的像素值
        img = self.pic_resize(img, (h, w), preserve=0)
        std_shell = np.load(os.path.join(os.path.dirname(__file__), 'std_shell.npy'))  # 标准扇贝轮廓
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转为HSV图像格式 HSV[:, :, 1] 得到图像的饱和度灰度图
        thres = cv2.threshold(HSV[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # 二值化，大津自动阈值
        kernel = np.ones((3, 3), np.uint8)  # 运算核
        mb = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel, iterations=3)  # 闭运算
        cnts = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 按照面积由大到小排序
        count = []
        for c in cnts:
            similar = 1 - cv2.matchShapes(std_shell, c, cv2.CONTOURS_MATCH_I1, 0.0)
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
                _, _, f, _ = defects[i, 0]
                # start = tuple(approx[s][0])
                # end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                p1.append(approx[f][0])
                # cv2.circle(img, far, 2, (0, 255, 255), -1)  # 缺陷点标注，草地绿
            x, y = int(p1[1][0] - (p1[1][0] - p1[0][0]) / 2), int(p1[1][1] - (p1[1][1] - p1[0][1]) / 2)  # 腰点中点的坐标
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)  # 缺陷点的中点，草地绿
            cv2.drawContours(img, [c], -1, (0, 0, 255), 2)  # 绘制轮廓线
            M = cv2.moments(c)  # 计算轮廓的矩，Hu_M = cv2.HuMoments(M)  # 计算7个不变矩
            if M['m00'] < 1000:
                break  # 面积小于1000个点的抛弃掉
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])  # 质心的坐标位置
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)  # 画质心，纯蓝色
            circle = cv2.minEnclosingCircle(c)  # 查找最小外接圆，因为如果是扇贝，圆心和质心差距应该不大，可以忽略，找外接圆的主要目的是确定画直线段的长度
            circle_r = int(circle[1])  # 半径
            line_2p = self.getcrosspoint(c, (cx, cy), (x, y), circle_r, 2)  # 返回A,B,C,D四个点
            cv2.line(img, line_2p[0], line_2p[1], (0, 255, 255), 2)  # 颜色 浅紫色
            cv2.circle(img, line_2p[0], 4, (255, 255, 0), -1)  # 2个端点
            cv2.circle(img, line_2p[1], 4, (255, 255, 0), -1)
            cv2.line(img, line_2p[2], line_2p[3], (200, 200, 0), 2)
            cv2.circle(img, line_2p[2], 4, (0, 255, 255), -1)  # 2个端点
            cv2.circle(img, line_2p[3], 4, (0, 255, 255), -1)
            shell_h, shell_w = self.linelength(line_2p[0], line_2p[1]) * self.rate, self.linelength(line_2p[2], line_2p[3]) * self.rate
            cv2.putText(img, 'No:{0}'.format(len(count) + 1), (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
            cv2.putText(img, 'H:{0:.1f}L:{1:.1f}'.format(shell_h, shell_w), (cx, cy + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
            cv2.putText(img, 'S:{:.1%}'.format(similar), (cx, cy + 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255))
            count.append([shell_h, shell_w, similar, M['m00']/100 * self.rate**2])  # 检测到这里，可以认为找到了扇贝，计数器可以加一了
            if len(count) >= maxnum:
                break
        return count, img

    def getcrosspoint(self, c, p_a, p_b, circle_c, iteration=1):
        '''根据给定的a,b两点和最小外接圆半径，求得和C形状相接的2点坐标，最后一个参数是迭代次数，一般不用超过3次。'''
        dis_s, dis_t = [circle_c for _ in range(4)], [0, 0, 0, 0]
        line_ret = self.getlinepoint(p_a, p_b, dis_s[0], dis_s[1], dis_s[2], dis_s[3])  # 返回cd两点
        for _ in range(iteration):
            for i in range(4):
                dis_t[i] = int(cv2.pointPolygonTest(c, line_ret[i], 1))  # 依次计算质点轴方向上点下点、垂直轴方向左侧右侧距离
                dis_s[i] = dis_s[i] + dis_t[i]
            line_ret = self.getlinepoint(p_a, p_b, dis_s[0], dis_s[1], dis_s[2], dis_s[3])
        return line_ret

    def getlinepoint(self, p_a, p_b, dis_c, dis_d, dis_e, dis_f):
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

    def pic_resize(self, img, newsize, preserve=1):  # 调整尺寸，第三个参数是是否保持比例，默认保持，若保持时，newsize的第二个值起作用
        h, w = img.shape[:2]
        h1, w1 = newsize
        if preserve == 1:
            #w1 = int(h1 * w / h)
            h1 = int(w1 * h /w)
        dst = cv2.resize(img, (w1, h1))
        return dst

    def rotate_bound(self, img, angle=90):  # 旋转任意角度
        h, w = img.shape[:2]  # 图像尺寸
        cX, cY = w // 2, h // 2  # 取图像中心
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(img, M, (nW, nH))

    def save_pic(self, img, path):
        cv2.imwrite(path, img)

    def rotate_90(self, img):  # 旋转90度
        return np.rot90(img).copy()

    def auto_detect(self, img=None, maxwidth=800, drawtxt=1, drawline=1):  # 根据给定的扇贝图片，自动识别纸张、扇贝
        '''返回值：正确 1, 扇贝最终图，扇贝数据列表， 0, img, None 表明只识别到扇贝纸张，未识别到扇贝数据，发回纸张图， -1, 原图像, None，完全未识别'''
        if img == None:
            if self.img.any() != None:
                img = self.img
            else:
                return -2, None, None
        h, w = img.shape[:2]  # 图像尺寸
        if w > 1000:
            img = self.pic_resize(img, (0, 1000))  # 最大边的值
            h, w = img.shape[:2]
        ret, approx, box = self.detect_rect(img)
        if not ret:
            if h > w:
                img = self.rotate_90(img)
            img = self.pic_resize(img, (0, maxwidth))
            h, w = img.shape[:2]
            cv2.putText(img, 'No A4 Paper Detected!', (w // 2, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            return -1, img, None  # 没有检测到白纸，直接返回完全未识别
        else:
            dst = self.transform(img, approx, box)  # 进行变形操作
            h, w = dst.shape[:2]
            if h > w:
                dst = self.rotate_90(dst)
                h, w = dst.shape[:2]  # 如果是竖着的，改成横向
            sd, final = self.detect_shell(dst)  # 检测扇贝
            if len(sd) <= 0:
                dst = self.pic_resize(dst, (0, maxwidth))  # 进行图片缩放
                h, w = dst.shape[:2]
                cv2.putText(dst, 'No Scallop Detected!', (w // 2, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                return 0, dst, None  # 没有检测到扇贝，返回修正后的白纸照片
            else:
                rate = 297 / w * 0.5 + 210 / h * 0.5
                final = self.pic_resize(final, (0, maxwidth))
                # sd = np.array(sd)
                # sd[:, [0, 1]] = sd[:, [0, 1]] * rate  # 前2列高和宽
                # sd[:, 3] = sd[:, 3] * rate ** 2 / 100  # 面积
                if drawtxt == 1:
                    xx, yy = 10, 20
                    for index, value in enumerate(sd):
                        text = ': H={0:.1f}mm, L={1:.1f}mm, S={2:.1%}, Area={3:.1f}cm2'.format(value[0], value[1],
                                                                                               value[2], value[3])
                        cv2.putText(final, str(index + 1) + text, (xx, yy), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0))
                        yy += 25
                    cv2.putText(final, 'Paper Size: {0:.1f}*{1:.1f}'.format(w * rate, h * rate), (xx, yy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                                (255, 0, 0))
                return 1, final, sd

if __name__ == '__main__':
    # shells = GetShellSize()
    # fn = 'timg_33.jpg'
    # img = cv2.imread('./pic/' + fn)
    # ret, show, sd = shells.auto_detect(img)
    shells = GetShellSize('./pic/timg_34.jpg')
    _, show, _ = shells.auto_detect()
    cv2.imshow('show', show)
    cv2.waitKey()
    cv2.destroyAllWindows()
