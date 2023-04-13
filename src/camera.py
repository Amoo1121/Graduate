import os
import datetime
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
                            QObject, QPoint, QRect, QSize, QTimer, QUrl, Qt, Signal, Slot, QEventLoop, SIGNAL)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
                           QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
                           QPixmap, QRadialGradient, QImage)
from PySide2.QtWidgets import *
import cv2
from pypylon import pylon


# class camera_update(QObject):
#     imgShow = Signal(QPixmap)
#
#     def flush(self):
#         pass


class ChildWindow():
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        self.ui = QUiLoader().load('./Resources/UI/camera.ui')
        # self.ui.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.center()
        self.camera_num = 0
        self.interval = 0
        self.timer_shot = QTimer()
        file_dir = os.getcwd()
        self.SaveFilePath = file_dir + '/Image'
        # self.dis_update = camera_update()
        # self.ui.camera_img.connect(self.dis_update, SIGNAL("imgShow(QPixmap)"), self.show)
        # self.ui.camera_img.connect(self.show)
        self.ui.SAVE_PATH.clicked.connect(self.FilePath)
        self.ui.MANUAL_SHOT.clicked.connect(self.ManualShot)
        self.ui.INTERVAL_SET.clicked.connect(self.IntervalSet)
        self.ui.AUTO_SHOT.clicked.connect(self.AutoShot)
        self.ui.CAMERA_SELECT.buttonClicked.connect(self.handleButtonClicked)
        self.ui.LAUNCH_CAMERA.clicked.connect(self.LaunchCamera)
        self.ui.STOP_CAMERA.clicked.connect(self.StopCamera)

    # @Slot()
    # def show(self, pixmap):
    #     self.ui.camera_img.setPixmap(pixmap)  # 把图像设置为背景
    #     self.ui.camera_img.setScaledContents(True)

    def center(self):
        # 窗口中心化
        qRect = self.ui.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qRect.moveCenter(centerPoint)
        self.ui.move(qRect.topLeft())

    def handleButtonClicked(self):
        camera_label = self.ui.CAMERA_SELECT.checkedButton().text()
        # print(camera_num)
        if camera_label == 'Basler相机':
            self.camera_num = 0
        elif camera_label == ' 海康威视相机':
            self.camera_num = 1
        elif camera_label == '电脑相机':
            self.camera_num = 2

    def FilePath(self):
        self.SaveFilePath = QFileDialog.getExistingDirectory(self.ui.SAVE_PATH, "请选择图像保存路径")  # 打开存储路径
        print('图像保存路径：{0}'.format(self.SaveFilePath))

    def SaveImage(self):
        i = datetime.datetime.now()
        time = '{0}-{1}-{2}-{3}_{4}_{5}_{6}'.format(i.year, i.month, i.day, i.hour, i.minute, i.second, i.microsecond)
        # print(self.SaveFilePath + '/' + time + '.jpg')
        file_name = self.SaveFilePath + '/' + time + '.jpg'
        # cv2.imshow(file_name,self.image)
        cv2.imwrite(file_name, self.image)
        # print(file_name)

    def ManualShot(self):
        self.SaveImage()

    def IntervalSet(self):
        self.interval = int(self.ui.INTERVAL.text())
        if self.interval > 0:
            self.ui.AUTO_SHOT.setEnabled(True)
            self.ui.INTERVAL_SET.setEnabled(False)
            self.ui.INTERVAL.setEnabled(False)
        else:
            QMessageBox.warning(self.ui.INTERVAL_SET, '警告', '请将间隔时间设置为大于0的数字！', QMessageBox.Yes)

    def AutoShot(self):
        self.timer_shot.timeout.connect(self.SaveImage)
        self.timer_shot.start(self.interval)
        self.ui.INTERVAL_SET.setEnabled(False)
        self.ui.INTERVAL.setEnabled(False)

    def LaunchCamera(self):
        if self.camera_num == 0:
            try:
                # connecting to the first available camera
                self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            except:
                QMessageBox.warning(self.ui.LAUNCH_CAMERA, '警告', '相机启动失败', QMessageBox.Yes)
            else:
                self.ui.MANUAL_SHOT.setEnabled(True)
                self.ui.STOP_CAMERA.setEnabled(True)
                self.ui.INTERVAL_SET.setEnabled(True)
                self.ui.INTERVAL.setEnabled(True)
                self.ui.LAUNCH_CAMERA.setEnabled(False)
                # Grabbing Continue (video) with minimal delay
                self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.timer_basler = QTimer()  # 设置定时器
                self.timer_basler.timeout.connect(self.basler_show)  # 这里调用不能有函数括号，不是单纯的运行函数
                self.timer_basler.start(5)  # 更新时间为每5ms
        if self.camera_num == 1:
            try:
                self.cap = cv2.VideoCapture("rtsp://username:passport@ip:port/Streaming/Channels/1")
            except:
                QMessageBox.warning(self.ui.LAUNCH_CAMERA, '警告', '相机启动失败', QMessageBox.Yes)
            else:
                self.ui.MANUAL_SHOT.setEnabled(True)
                self.ui.STOP_CAMERA.setEnabled(True)
                self.ui.INTERVAL_SET.setEnabled(True)
                self.ui.INTERVAL.setEnabled(True)
                self.ui.LAUNCH_CAMERA.setEnabled(False)
                # Grabbing Continue (video) with minimal delay
                self.timer_hik = QTimer()  # 设置定时器
                self.timer_hik.timeout.connect(self.hik_show)  # 这里调用不能有函数括号，不是单纯的运行函数
                self.timer_hik.start(5)  # 更新时间为每5ms
        if self.camera_num == 2:
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except:
                QMessageBox.warning(self.ui.LAUNCH_CAMERA, '警告', '相机启动失败', QMessageBox.Yes)
            else:
                self.ui.MANUAL_SHOT.setEnabled(True)
                self.ui.STOP_CAMERA.setEnabled(True)
                self.ui.INTERVAL_SET.setEnabled(True)
                self.ui.INTERVAL.setEnabled(True)
                self.ui.LAUNCH_CAMERA.setEnabled(False)
                # Grabbing Continue (video) with minimal delay
                self.timer_pc = QTimer()  # 设置定时器
                self.timer_pc.timeout.connect(self.pc_show)  # 这里调用不能有函数括号，不是单纯的运行函数
                self.timer_pc.start(5)  # 更新时间为每5ms

    def StopCamera(self):
        if self.camera_num == 0:
            self.timer_shot.stop()
            self.timer_basler.stop()
            self.cam.StopGrabbing()
            self.cam.Close()
            self.ui.MANUAL_SHOT.setEnabled(False)
            self.ui.AUTO_SHOT.setEnabled(False)
            self.ui.STOP_CAMERA.setEnabled(False)
            self.ui.INTERVAL_SET.setEnabled(False)
            self.ui.INTERVAL.setEnabled(False)
            self.ui.LAUNCH_CAMERA.setEnabled(True)
            self.ui.camera_img.setPixmap('')  # 还原label
            self.ui.camera_img.setScaledContents(True)
            os.startfile(self.SaveFilePath)
        elif self.camera_num == 1:
            self.timer_shot.stop()
            self.timer_pc.stop()
            self.cap.release()
            self.ui.MANUAL_SHOT.setEnabled(False)
            self.ui.AUTO_SHOT.setEnabled(False)
            self.ui.STOP_CAMERA.setEnabled(False)
            self.ui.INTERVAL_SET.setEnabled(False)
            self.ui.INTERVAL.setEnabled(False)
            self.ui.LAUNCH_CAMERA.setEnabled(True)
            self.ui.camera_img.setPixmap('')  # 还原label
            self.ui.camera_img.setScaledContents(True)
            os.startfile(self.SaveFilePath)
        elif self.camera_num == 2:
            self.timer_shot.stop()
            self.timer_pc.stop()
            self.cap.release()
            self.ui.MANUAL_SHOT.setEnabled(False)
            self.ui.AUTO_SHOT.setEnabled(False)
            self.ui.STOP_CAMERA.setEnabled(False)
            self.ui.INTERVAL_SET.setEnabled(False)
            self.ui.INTERVAL.setEnabled(False)
            self.ui.LAUNCH_CAMERA.setEnabled(True)
            self.ui.camera_img.setPixmap('')  # 还原label
            self.ui.camera_img.setScaledContents(True)
            os.startfile(self.SaveFilePath)

    def basler_show(self):
        converter = pylon.ImageFormatConverter()
        # converting to opencv bgr format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        if self.cam.IsGrabbing():
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = self.cam.RetrieveResult(200, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # image = converter.Convert(grabResult)
                # image = image.GetArray()
                self.image = grabResult.Array
                cv2.imwrite('./1.jpg', self.image)
                img_file = cv2.imread('./1.jpg')
                self.ui.camera_img.setPixmap('./1.jpg')  # 把图像设置为背景
                self.ui.camera_img.setScaledContents(True)
                # img = QImage(self.image, self.image.shape[1], self.image.shape[0],
                #              QImage.Format_RGB888)  # Qlmage的参数（data, width, height, bytesPerLine, format ）图像存储使用8-8-8 24位RGB格式
                # pixmap = QPixmap.fromImage(image)
                # pixmap = QPixmap(img)
                # self.dis_update.imgShow.emit(pixmap)
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()

    def hik_show(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转BGR，不然直接显示会变蓝
        self.image = frame
        img = QImage(frame, frame.shape[1], frame.shape[0],
                     frame.strides[0],
                     QImage.Format_RGB888)  # Qlmage的参数（data, width, height, bytesPerLine, format ）图像存储使用8-8-8 24位RGB格式
        self.ui.camera_img.setPixmap(QPixmap.fromImage(img))  # 把图像设置为背景
        self.ui.camera_img.setScaledContents(True)

    def pc_show(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
        self.image = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB转BGR，不然直接显示会变蓝
        img = QImage(frame, frame.shape[1], frame.shape[0],
                     frame.strides[0],
                     QImage.Format_RGB888)  # Qlmage的参数（data, width, height, bytesPerLine, format ）图像存储使用8-8-8 24位RGB格式
        self.ui.camera_img.setPixmap(QPixmap.fromImage(img))  # 把图像设置为背景
        self.ui.camera_img.setScaledContents(True)
