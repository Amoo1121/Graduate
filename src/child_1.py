from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QDesktopWidget
from PySide2.QtUiTools import QUiLoader
import PySide2.QtGui as QtGui
# from PySide2 import QtCore
from scipy import ndimage
import scipy.signal as signal
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pywt import dwt2, idwt2
import os
import sys
# import screen


class ChildWindow:
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        self.rimage = None
        self.backup1 = None
        self.backup2 = None
        self.ui = QUiLoader().load('./Resources/UI/child_1.ui')
        # self.ui.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.center()
        self.ui.chooseFile.clicked.connect(self.openFileNameDialog)
        self.ui.save.clicked.connect(self.saveFile)
        self.ui.backout.clicked.connect(self.backout)
        self.ui.resButton.clicked.connect(self.reset_app)
        self.ui.stopButton.clicked.connect(self.exit_app)
        self.ui.grey.clicked.connect(self.grey_trans)
        self.ui.Trans_radon.clicked.connect(self.Trans_radon)
        self.ui.G_Trans.clicked.connect(self.G_Trans)
        self.ui.erode.clicked.connect(self.erode)
        self.ui.dilate.clicked.connect(self.dilate)
        self.ui.dilate.clicked.connect(self.smooth)
        self.ui.filter.clicked.connect(self.filter)
        self.ui.sharpen.clicked.connect(self.sharpen)
        self.ui.FFT.clicked.connect(self.FFT)
        self.ui.lowPassFiltering.clicked.connect(self.lowPassFiltering)
        self.ui.highPassFiltering.clicked.connect(self.highPassFiltering)
        self.ui.DWT.clicked.connect(self.DWT)  # 以上均为点击按钮后连接的功能函数
        # self.ui.logo.setPixmap('./Resources/Icon/a_91tv.jpg')

    def center(self):
        # 窗口中心化
        qRect = self.ui.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qRect.moveCenter(centerPoint)
        self.ui.move(qRect.topLeft())

    def exit_app(self):
        self.ui.close()

    def backout(self):
        self.rimage = self.backup1
        self.backup2 = self.backup1
        img = QtGui.QImage(self.backup1, self.backup1.shape[1], self.backup1.shape[0], QtGui.QImage.Format_BGR888)
        img = QtGui.QPixmap(img)
        self.ui.rImage.setPixmap(img)
        self.ui.rImage.setScaledContents(True)

    def reset_app(self):
        self.ui.oImage.setPixmap(self.image_file)  #重置图片
        self.ui.oImage.setScaledContents(True) #使图片尺寸适应控件大小
        self.ui.rImage.setPixmap(self.image_file)
        self.ui.rImage.setScaledContents(True)
        self.ui.pImage.setPixmap(self.image_file)
        self.ui.pImage.setScaledContents(True)
        self.rimage = cv2.imread(self.image_file)
        self.backup1 = self.rimage
        self.backup2 = self.rimage

    def openFileNameDialog(self):
        FileDialog = QFileDialog(self.ui.chooseFile)
        # 设置可以打开任何文件
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        Filter = "(*.jpg,*.png,*.jpeg,*.bmp,*.gif)|*.jgp;*.png;*.jpeg;*.bmp;*.gif|All files(*.*)|*.*"
        self.image_file, _ = FileDialog.getOpenFileName(self.ui.chooseFile, 'open file', './',
                                                   'Image files (*.jpg *.gif *.png *.jpeg)')  # 选择目录，返回选中的路径 'Image files (*.jpg *.gif *.png *.jpeg)'
        # 判断是否正确打开文件
        if not self.image_file:
            QMessageBox.warning(self.ui.chooseFile, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return
        self.ui.oImage.setPixmap(self.image_file)
        self.ui.oImage.setScaledContents(True)
        self.ui.rImage.setPixmap(self.image_file)
        self.ui.rImage.setScaledContents(True)
        self.ui.pImage.setPixmap(self.image_file)
        self.ui.pImage.setScaledContents(True)
        self.rimage = cv2.imread(self.image_file)
        self.backup1 = self.rimage
        self.backup2 = self.rimage

    def saveFile(self):
        SaveFilePath = QFileDialog.getExistingDirectory(self.ui.save) #打开存储路径
        cv2.imwrite(os.path.join(SaveFilePath, 'Processed image.png'), self.rimage) #保存图像
        # cv2.imwrite(os.path.join(SaveFilePath, 'Pre-processed image.png'), pimage)
        # cv2.imwrite('预处理图像', pimage)

    def grey_trans(self):
        # self.ui.pImage.setText("无")
        img = self.rimage
        value_max = np.max(img)
        y = value_max - img
        self.rimage = y
        self.backup1 = self.backup2
        self.backup2 = y
        # 将图片转化成Qt可读格式
        grey_image = QtGui.QImage(y, y.shape[1], y.shape[0], QtGui.QImage.Format_BGR888)
        grey_image = QtGui.QPixmap(grey_image)
        self.ui.rImage.setPixmap(grey_image)
        self.ui.rImage.setScaledContents(True)

    def Trans_radon(self):
        # self.ui.pImage.setText("无")
        opt = self.ui.Trans_Opt.currentIndex()
        if opt == 0:
            img = Image.fromarray(self.rimage)
            img = ImageEnhance.Contrast(img).enhance(3)  # 对比度增强类,用于调整图像的对比度,3为增强3倍
            img = np.array(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # phough_image = QtGui.QImage(gray, gray.shape[1], gray.shape[0], QtGui.QImage.Format_BGR888)
            # phough_image = QtGui.QPixmap(phough_image)
            # self.ui.pImage.setPixmap(phough_image)
            # self.ui.pImage.setScaledContents(True)
            ret, binary = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_OTSU)  # 阈值变换，cv2.THRESH_OTSU适合用于双峰值图像,ret是阈值，binary是变换后的图像
            # canny边缘检测
            edges = cv2.Canny(binary, ret - 30, ret + 30, apertureSize=3)  # 图像，最小阈值，最大阈值，sobel算子的大小
            lines = cv2.HoughLinesP(edges, 1, 1 * np.pi / 180, 10, minLineLength=10,
                                    maxLineGap=5)  # 统计概率霍夫线变换函数：图像矩阵，极坐标两个参数，一条直线所需最少的曲线交点，组成一条直线的最少点的数量，被认为在一条直线上的亮点的最大距离
            self.ui.textEdit.setText("Line Num : {0}".format(len(lines)))
            # 画出检测的线段
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                pass
            self.rimage = img
            self.backup1 = self.backup2
            self.backup2 = img
            hough_image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_BGR888)
            hough_image = QtGui.QPixmap(hough_image)
            self.ui.rImage.setPixmap(hough_image)
            self.ui.rImage.setScaledContents(True)
            # print(type(rimage))
        elif opt == 1:
            # image = cv2.cvtColor(rimage, cv2.COLOR_RGB2GRAY)
            image = Image.fromarray(self.rimage)
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            magnitude_spectrum = np.ascontiguousarray(magnitude_spectrum)
            self.rimage = magnitude_spectrum
            self.backup1 = self.backup2
            self.backup2 = magnitude_spectrum
            f_image = QtGui.QImage(magnitude_spectrum, magnitude_spectrum.shape[1], magnitude_spectrum.shape[0],
                                   QtGui.QImage.Format_BGR888)
            f_image = QtGui.QPixmap(f_image)
            self.ui.rImage.setPixmap(f_image)
            self.ui.rImage.setScaledContents(True)

    def G_Trans(self):
        # 获取原始图像列数和行数
        rows, cols, channel = self.rimage.shape
        result = self.rimage
        delta_x = self.ui.delta_x.text()
        delta_y = self.ui.delta_y.text()
        delta_degree = int(self.ui.delta_degree.text())
        if self.ui.HMcheckBox.isChecked():
            result = cv2.flip(result, 1)
        if self.ui.VMcheckBox.isChecked():
            result = cv2.flip(result, 0)
        if delta_degree != 0:
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), delta_degree, 1)
            # 函数参数：原始图像 旋转参数 元素图像宽高
            result = cv2.warpAffine(result, M, (cols, rows))
        if delta_x != 0 and delta_y != 0:
            # 图像平移矩阵
            M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
            # 图像平移
            result = cv2.warpAffine(result, M, (cols, rows))
        self.rimage = result
        self.backup1 = self.backup2
        self.backup2 = result
        GTrans_image = QtGui.QImage(result, result.shape[1], result.shape[0], QtGui.QImage.Format_BGR888)
        GTrans_image = QtGui.QPixmap(GTrans_image)
        self.ui.rImage.setPixmap(GTrans_image)
        self.ui.rImage.setScaledContents(True)

    def erode(self):
        kernel_size = int(self.ui.ED_KERNEL_SIZE.text())
        # 设置腐蚀和膨胀核
        kernel = np.ones(shape=[kernel_size, kernel_size], dtype=np.uint8)  # 通过shape=[3,3]可以改变处理效果
        OriginErodeImg = cv2.erode(self.rimage, kernel=kernel)
        self.rimage = OriginErodeImg
        self.backup1 = self.backup2
        self.backup2 = OriginErodeImg
        erode_image = QtGui.QImage(OriginErodeImg, OriginErodeImg.shape[1], OriginErodeImg.shape[0],
                                   QtGui.QImage.Format_BGR888)
        erode_image = QtGui.QPixmap(erode_image)
        self.ui.rImage.setPixmap(erode_image)
        self.ui.rImage.setScaledContents(True)

    def dilate(self):
        kernel_size = int(self.ui.ED_KERNEL_SIZE.text())
        # 设置腐蚀和膨胀核
        kernel = np.ones(shape=[kernel_size, kernel_size], dtype=np.uint8)  # 通过shape=[3,3]可以改变处理效果
        img = cv2.dilate(self.rimage, kernel, iterations=1)
        self.rimage = img
        self.backup1 = self.backup2
        self.backup2 = img
        dilate_image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_BGR888)
        dilate_image = QtGui.QPixmap(dilate_image)
        self.ui.rImage.setPixmap(dilate_image)
        self.ui.rImage.setScaledContents(True)

    def smooth(self):
        KERNEL_SIZE = int(self.ui.FILTER_KERNEL_SIZE.text())
        img = self.rimage
        img = cv2.blur(img, (KERNEL_SIZE, KERNEL_SIZE))
        self.rimage = img
        self.backup1 = self.backup2
        self.backup2 = img
        smooth_image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_BGR888)
        smooth_image = QtGui.QPixmap(smooth_image)
        self.ui.rImage.setPixmap(smooth_image)
        self.ui.rImage.setScaledContents(True)

    def filter(self):
        KERNEL_SIZE = int(self.ui.FILTER_KERNEL_SIZE.text())
        img = cv2.medianBlur(self.rimage, KERNEL_SIZE)
        self.rimage = img
        self.backup1 = self.backup2
        self.backup2 = img
        smooth_image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_BGR888)
        smooth_image = QtGui.QPixmap(smooth_image)
        self.ui.rImage.setPixmap(smooth_image)
        self.ui.rImage.setScaledContents(True)

    def sharpen(self):
        img = cv2.addWeighted(self.rimage, 2, cv2.GaussianBlur(self.rimage, (0, 0), 10), -1, 128)
        self.rimage = img
        self.backup1 = self.backup2
        self.backup2 = img
        sharpen_image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_BGR888)
        sharpen_image = QtGui.QPixmap(sharpen_image)
        self.ui.rImage.setPixmap(sharpen_image)
        self.ui.rImage.setScaledContents(True)

    def FFT(self):
        image = Image.fromarray(self.rimage)
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        magnitude_spectrum = np.ascontiguousarray(magnitude_spectrum)
        self.rimage = magnitude_spectrum
        self.backup1 = self.backup2
        self.backup2 = magnitude_spectrum
        f_image = QtGui.QImage(magnitude_spectrum, magnitude_spectrum.shape[1], magnitude_spectrum.shape[0],
                               QtGui.QImage.Format_BGR888)
        f_image = QtGui.QPixmap(f_image)
        self.ui.rImage.setPixmap(f_image)
        self.ui.rImage.setScaledContents(True)

    def lowPassFiltering(self):
        img = cv2.cvtColor(self.rimage, cv2.COLOR_BGR2GRAY)
        # 傅里叶变换
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(dft)
        res = fshift
        # 设置低通滤波器
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
        # 掩膜图像和频谱图像乘积
        f = fshift * mask
        # 傅里叶逆变换
        ishift = np.fft.ifftshift(f)
        iimg = cv2.idft(ishift)
        res_img = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

        res_img = np.ascontiguousarray(res_img)
        r_image = QtGui.QImage(res_img, res_img.shape[1], res_img.shape[0], QtGui.QImage.Format_BGR888)
        r_image = QtGui.QPixmap(r_image)
        self.ui.rImage.setPixmap(r_image)
        self.ui.rImage.setScaledContents(True)
        res = np.ascontiguousarray(res)
        p_image = QtGui.QImage(res, res.shape[1], res.shape[0], QtGui.QImage.Format_BGR888)
        p_image = QtGui.QPixmap(p_image)
        self.ui.pImage.setPixmap(p_image)
        self.ui.pImage.setScaledContents(True)

    def highPassFiltering(self):
        img = self.rimage
        # 傅里叶变换
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        res = fshift
        # 设置高通滤波器
        rows, cols, channel = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        # 傅里叶逆变换
        ishift = np.fft.ifftshift(fshift)
        res_img = np.fft.ifft2(ishift)
        res_img = np.abs(res_img)

        res_img = np.ascontiguousarray(res_img)
        r_image = QtGui.QImage(res_img, res_img.shape[1], res_img.shape[0], QtGui.QImage.Format_BGR888)
        r_image = QtGui.QPixmap(r_image)
        self.ui.rImage.setPixmap(r_image)
        self.ui.rImage.setScaledContents(True)
        res = np.ascontiguousarray(res)
        p_image = QtGui.QImage(res, res.shape[1], res.shape[0], QtGui.QImage.Format_BGR888)
        p_image = QtGui.QPixmap(p_image)
        self.ui.pImage.setPixmap(p_image)
        self.ui.pImage.setScaledContents(True)

    def DWT(self):
        img = cv2.cvtColor(self.rimage, cv2.COLOR_BGR2GRAY)
        # 对img进行haar小波变换：
        cA, (cH, cV, cD) = dwt2(img, 'haar')
        # # 小波变换之后，低频分量对应的图像：
        # cv2.imwrite('lena.png', np.uint8(cA / np.max(cA) * 255))
        # # 小波变换之后，水平方向高频分量对应的图像：
        # cv2.imwrite('lena_h.png', np.uint8(cH / np.max(cH) * 255))
        # # 小波变换之后，垂直平方向高频分量对应的图像：
        # cv2.imwrite('lena_v.png', np.uint8(cV / np.max(cV) * 255))
        # # 小波变换之后，对角线方向高频分量对应的图像：
        # cv2.imwrite('lena_d.png', np.uint8(cD / np.max(cD) * 255))
        # 根据小波系数重构回去的图像
        res_img = idwt2((cA, (cH, cV, cD)), 'haar')

        cv2.imwrite('rimg.png', np.uint8(res_img))
        r_image = QtGui.QImage(res_img, res_img.shape[1], res_img.shape[0], QtGui.QImage.Format_BGR888)
        r_image = QtGui.QPixmap(r_image)
        self.ui.rImage.setPixmap(r_image)
        self.ui.rImage.setScaledContents(True)
