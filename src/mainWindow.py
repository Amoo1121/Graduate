from PySide2.QtWidgets import QApplication, QMessageBox, QDesktopWidget
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore
import sys
# import qdarkstyle
# import os
from qt_material import apply_stylesheet
from src import child_1, child_2, child_3, camera, console


class experiment:

    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        self.ui = QUiLoader().load('./Resources/UI/MainWindow.ui')
        self.ui.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.ui.pushButton.clicked.connect(self.camera)
        self.ui.button_1.clicked.connect(self.baseExp)
        self.ui.button_2.clicked.connect(self.advanceExp)
        self.ui.button_3.clicked.connect(self.nngenerator)
        self.ui.console_info.clicked.connect(self.CONSOLE_INFO)
        self.ui.actiona.triggered.connect(self.exit_app)
        self.ui.about.triggered.connect(self.about_app)
        self.camera = camera.ChildWindow()
        self.childwindow_1 = child_1.ChildWindow()
        self.childwindow_2 = child_2.ChildWindow()
        self.childwindow_3 = child_3.ChildWindow()
        self.childwindow_4 = console.ChildWindow()
        self.childwindow_4.ui.show()
        self.center()

    def center(self):
        # 窗口中心化
        qRect = self.ui.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qRect.moveCenter(centerPoint)
        self.ui.move(qRect.topLeft())

    def camera(self):
        self.camera.ui.show()

    def baseExp(self):
        self.childwindow_1.ui.show()

    def advanceExp(self):
        self.childwindow_2.ui.show()

    def nngenerator(self):
        self.childwindow_3.ui.show()

    def CONSOLE_INFO(self):
        self.childwindow_4.ui.show()

    def exit_app(self):
        self.camera.ui.close();
        self.childwindow_1.ui.close();
        self.childwindow_2.ui.close();
        self.childwindow_3.ui.close();
        self.childwindow_4.ui.close();
        self.ui.close()

    def about_app(self):
        QMessageBox.about(self.ui,
                          '关于',
                          '1111'
                          )


app = QApplication(sys.argv)
# # setup stylesheet
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyside2())
# # or in new API
# app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside2'))
apply_stylesheet(app, theme='dark_teal.xml')
expe = experiment()
expe.ui.show()  # 设置主题并运行
app.exec_()
sys.exit(0)
