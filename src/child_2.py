from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QDesktopWidget
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore, QtGui
from PySide2.QtCore import QEventLoop, QTimer
from PySide2.QtCore import Slot
import sys
import os
# import screen
import signal
import matplotlib.pyplot as plt

sys.path.append('./AE')
sys.path.append('./MNIST')
sys.path.append('./Fashion_MNIST')
import numpy as np
import cv2
import MNIST_TRAIN
import MNIST_TEST
import Fashion_MNIST_TEST
import Fashion_MNIST_TRAIN
import AE_TEST
import AE_TRAIN
from cfg import DefaultConfig


# class EmittingStr(QtCore.QObject):
#     textWritten = QtCore.Signal(str)  # 定义一个发送str的信号，这里用的方法名与PyQt5不一样
#
#     def write(self, text):
#         self.textWritten.emit(str(text))
#         loop = QEventLoop()
#         QTimer.singleShot(100, loop.quit)
#         loop.exec_()
#
#     def flush(self):
#         pass


class ChildWindow():
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        self.ui = QUiLoader().load('./Resources/UI/child_2.ui')
        self.center()
        # self.ui.Window_Clear.clicked.connect(self.Window_Clear)
        self.ui.Pre_MNIST_TEST.clicked.connect(self.Pre_MNIST)
        self.ui.MNIST_TRAIN.clicked.connect(self.MNIST_TRAIN)
        self.ui.MNIST_TEST.clicked.connect(self.MNIST_TEST)
        self.ui.MNIST_OPEN.clicked.connect(self.MNIST_OPEN)
        self.ui.MNIST_NETWORK_FILE.clicked.connect(self.MNIST_NetworkFile)
        self.ui.Pre_Fashion_MNIST_TEST.clicked.connect(self.Pre_Fashion_MNIST)
        self.ui.Fashion_MNIST_TRAIN.clicked.connect(self.Fashion_MNIST_TRAIN)
        self.ui.Fashion_MNIST_TEST.clicked.connect(self.Fashion_MNIST_TEST)
        self.ui.Fashion_MNIST_OPEN.clicked.connect(self.Fashion_MNIST_OPEN)
        self.ui.Fashion_MNIST_NETWORK_FILE.clicked.connect(self.Fashion_MNIST_NetworkFile)
        self.ui.Pre_AE_TEST.clicked.connect(self.Pre_AE_TEST)
        self.ui.AE_TEST.clicked.connect(self.AE_TEST)
        self.ui.AE_PROJECT_OPEN.clicked.connect(self.AE_PROJECT_OPEN)
        self.ui.AE_MODEL_OPEN.clicked.connect(self.AE_MODEL_OPEN)
        self.ui.AE_CHOOSE.clicked.connect(self.AE_CHOOSE)
        self.ui.VIDEO_OPEN.clicked.connect(self.VIDEO_OPEN)
        self.ui.IMG_SAVE.clicked.connect(self.IMG_SAVE)
        self.ui.convert.clicked.connect(self.convert)

        # sys.stdout = EmittingStr()
        # self.ui.edt_log.connect(sys.stdout, QtCore.SIGNAL("textWritten(QString)"), self.outputWritten)
        # sys.stderr = EmittingStr()
        # self.ui.edt_log.connect(sys.stderr, QtCore.SIGNAL("textWritten(QString)"), self.outputWritten)

    # @Slot()
    # def outputWritten(self, text):
    #     cursor = self.ui.edt_log.textCursor()
    #     cursor.movePosition(QtGui.QTextCursor.End)
    #     cursor.insertText(text)
    #
    #     self.ui.edt_log.setTextCursor(cursor)
    #     self.ui.edt_log.ensureCursorVisible()

    # def execCmd(self, cmd):
    #     r = os.popen(cmd)
    #     text = r.read()
    #     r.close()
    #     return text

    def center(self):
        qRect = self.ui.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qRect.moveCenter(centerPoint)
        self.ui.move(qRect.topLeft())

    # def Window_Clear(self):
    #     self.ui.edt_log.clear()

    def Pre_MNIST(self):
        file_dir = os.getcwd()
        # print(file_dir)
        MNIST_TEST.test(file_dir + '/MNIST/network_model')
        # # 画图
        # plt.plot(epoch_list, accuracy_list)
        # plt.xlabel('epoch')
        # plt.ylabel('accuracy')
        # plt.grid()
        # plt.show()

    def MNIST_TRAIN(self):
        file_dir = os.getcwd()
        epoch = int(self.ui.MNIST_EPOCH.text())
        MNIST_TRAIN.train(file_dir, epoch)

    def MNIST_TEST(self):
        file_dir = os.getcwd()
        # print(file_dir)
        MNIST_TEST.test(file_dir + '/MNIST/NEW_Network_Model')

    def MNIST_OPEN(self):
        file_dir = os.getcwd()
        if os.path.exists(file_dir + '/MNIST/NEW_Network_Model'):
            os.startfile(file_dir + '/MNIST/NEW_Network_Model')
        else:
            QMessageBox.warning(self.ui.MNIST_NETWORK_FILE, '警告', '文件夹不存在', QMessageBox.Yes)

    def MNIST_NetworkFile(self):
        file_dir = os.getcwd()
        os.startfile(file_dir + '/MNIST')

    def Pre_Fashion_MNIST(self):
        file_dir = os.getcwd()
        # print(file_dir)
        Fashion_MNIST_TEST.test(file_dir + '/Fashion_MNIST/network_model')

    def Fashion_MNIST_TRAIN(self):
        file_dir = os.getcwd()
        epoch = int(self.ui.Fashion_MNIST_EPOCH.text())
        Fashion_MNIST_TRAIN.train(file_dir, epoch)

    def Fashion_MNIST_TEST(self):
        file_dir = os.getcwd()
        # print(file_dir)
        Fashion_MNIST_TEST.test(file_dir + '/Fashion_MNIST/NEW_Network_Model')

    def Fashion_MNIST_OPEN(self):
        file_dir = os.getcwd()
        if os.path.exists(file_dir + '/Fashion_MNIST/NEW_Network_Model'):
            os.startfile(file_dir + '/Fashion_MNIST/NEW_Network_Model')
        else:
            QMessageBox.warning(self.ui.Fashion_MNIST_NETWORK_FILE, '警告', '文件夹不存在', QMessageBox.Yes)

    def Fashion_MNIST_NetworkFile(self):
        file_dir = os.getcwd()
        os.startfile(file_dir + '/Fashion_MNIST')

    def AE_PROJECT_OPEN(self):
        file_dir = os.getcwd()
        os.startfile(file_dir + '/AE')

    def AE_MODEL_OPEN(self):
        file_dir = os.getcwd()
        os.startfile(file_dir + '/AE/New_Network_Model')

    def Pre_AE_TEST(self):
        file_dir = os.getcwd()
        # print(file_dir)
        AE_TEST.test(file_dir + '/AE/network_model',self.AE_TEST_FILE)

    def AE_TEST(self):
        file_dir = os.getcwd()
        AE_TEST.test(file_dir + '/AE/New_Network_Model')

    def AE_CHOOSE(self):
        FileDialog = QFileDialog(self.ui.AE_CHOOSE)
        # 设置可以打开任何文件
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        Filter = "(*.jpg,*.png,*.jpeg,*.bmp,*.gif)|*.jgp;*.png;*.jpeg;*.bmp;*.gif|All files(*.*)|*.*"
        self.AE_TEST_FILE, _ = FileDialog.getOpenFileName(self.ui.AE_CHOOSE, '选择测试图像', './AE/data/bottle/test',
                                                        'Image files (*.jpg *.gif *.png *.jpeg)')  # 选择目录，返回选中的路径 'Image files (*.jpg *.gif *.png *.jpeg)'
        # 判断是否正确打开文件
        if not self.AE_TEST_FILE:
            QMessageBox.warning(self.ui.AE_CHOOSE, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return
        else:
            self.ui.Pre_AE_TEST.setEnabled(True)

    def VIDEO_OPEN(self):
        FileDialog = QFileDialog(self.ui.VIDEO_OPEN)
        # 设置可以打开任何文件
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # # 文件过滤
        # Filter = "(*.mov,*.avi,*.mp4,*.mpg,*.mpeg,*.m4v,*.mkv)|All files(*.*)|*.*"
        global video_file
        video_file, _ = FileDialog.getOpenFileName(self.ui.VIDEO_OPEN, '选择待转换媒体文件', './', )  # 选择目录，返回选中的路径
        # 判断是否正确打开文件
        if not video_file:
            QMessageBox.warning(self.ui.VIDEO_OPEN, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return
        else:
            videoCapture = cv2.VideoCapture(video_file)
            # 总帧数(frames)
            frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.ui.frame.setText(str(frames))

    def IMG_SAVE(self):
        global SaveFilePath
        SaveFilePath = QFileDialog.getExistingDirectory(self.ui.IMG_SAVE)  # 打开存储路径
        print('转换后文件存放路径：{0}'.format(SaveFilePath))

    def convert(self):
        global video_file, SaveFilePath
        if not video_file or not SaveFilePath:
            QMessageBox.warning(self.ui.convert, "警告", "未选择文件或存放路径！", QMessageBox.Yes)
            return
        else:
            frame_rate = int(self.ui.fps.text())
            videoCapture = cv2.VideoCapture(video_file)
            frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
            j = 0
            i = 0
            self.ui.progressBar.setRange(0, 100)  # 显示进度条
            self.ui.progressBar.setValue(0)
            # 读帧
            success, frame = videoCapture.read()
            while success:
                percentage = int((i / frames) * 100)
                self.ui.progressBar.setValue(percentage)
                i = i + 1
                # 每隔固定帧保存一张图片
                if i % frame_rate == 0:
                    j = j + 1
                    pic_address = SaveFilePath + '/' + str(j) + '.jpg'
                    cv2.imwrite(pic_address, frame)
                success, frame = videoCapture.read()
            self.ui.progressBar.setRange(0, 0)  # 重置进度条
            self.ui.progressBar.setValue(0)
            os.startfile(SaveFilePath)
