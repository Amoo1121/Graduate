from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QDesktopWidget, QButtonGroup
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore, QtGui
from PySide2.QtCore import QEventLoop, QTimer
from PySide2.QtCore import Slot
import sys
import os

class EmittingStr(QtCore.QObject):
    textWritten = QtCore.Signal(str)  # 定义一个发送str的信号，这里用的方法名与PyQt5不一样

    def write(self, text):
        self.textWritten.emit(str(text))
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec_()

    def flush(self):
        pass


class ChildWindow():
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        self.ui = QUiLoader().load('./Resources/UI/console.ui')
        # self.ui.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.center()
        self.ui.CLEAR_CONSOLE.clicked.connect(self.Window_Clear)

        sys.stdout = EmittingStr()
        self.ui.edt_log.connect(sys.stdout, QtCore.SIGNAL("textWritten(QString)"), self.outputWritten)
        sys.stderr = EmittingStr()
        self.ui.edt_log.connect(sys.stderr, QtCore.SIGNAL("textWritten(QString)"), self.outputWritten)

    @Slot()
    def outputWritten(self, text):
        cursor = self.ui.edt_log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.ui.edt_log.setTextCursor(cursor)
        self.ui.edt_log.ensureCursorVisible()

    def center(self):
        # 窗口中心化
        qRect = self.ui.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qRect.moveCenter(centerPoint)
        self.ui.move(qRect.topLeft())

    def Window_Clear(self):
        self.ui.edt_log.clear()
