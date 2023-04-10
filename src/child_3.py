from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QDesktopWidget
from PySide2.QtUiTools import QUiLoader
import PySide2.QtGui as QtGui
import screen
import os
import shutil
from string import Template


class ChildWindow:
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        self.ui = QUiLoader().load('./Resources/UI/child_3.ui')
        self.center()
        global lay_index, network, header_code, current_code
        lay_index = -1
        network = []
        current_code = '\t\tself.自定义网络名称 = nn.Sequential(\n'
        header_code = '\t\tself.自定义网络名称 = nn.Sequential(\n'
        # self.ui.textBrowser.setSource(current_code)
        self.ui.create_layer.clicked.connect(self.create_layer)
        self.ui.delete_layer.clicked.connect(self.delete_layer)
        self.ui.comboBox.currentIndexChanged.connect(self.edit_disable)
        self.ui.add_conv.clicked.connect(self.add_conv)
        self.ui.add_activation.clicked.connect(self.add_activation)
        self.ui.add_nor.clicked.connect(self.add_nor)
        self.ui.add_pool.clicked.connect(self.add_pool)
        self.ui.add_dropout.clicked.connect(self.add_dropout)
        self.ui.complete_network.clicked.connect(self.complete)
        self.ui.clear_all.clicked.connect(self.clear)
        self.ui.generate.clicked.connect(self.generate)

    def center(self):
        # 窗口中心化
        qRect = self.ui.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qRect.moveCenter(centerPoint)
        self.ui.move(qRect.topLeft())

    def edit_disable(self):
        if self.ui.comboBox.currentIndex() == 2:
            self.ui.kernel_size.setEnabled(False)
            self.ui.stride.setEnabled(False)
            self.ui.padding.setEnabled(False)
        else:
            self.ui.kernel_size.setEnabled(True)
            self.ui.stride.setEnabled(True)
            self.ui.padding.setEnabled(True)

    def exit_app(self):
        self.ui.close()

    def create_layer(self):
        global lay_index, current_code, header_code, network
        lay_index += 1
        if lay_index >= 1:
            len_code = len(current_code)
            if current_code[len_code - 1] == '\n':
                current_code += ')'  # 判断是否要加最后的括号，如果不判断再多次删除层之后会导致括号变多
            network[lay_index - 1] = current_code  # 更新network
            network.append(header_code)
            current_code = header_code
        else:
            network.append(header_code)
            current_code = header_code
        self.display()

    def delete_layer(self):
        global lay_index, network, current_code
        if lay_index > 0:
            current_code = network[lay_index - 1]  # 整体的变量的逻辑的问题，必须在删除时将当前代码替换成上一层的代码
            lenth = len(network)
            network.pop(lenth - 1)
            lay_index = lenth - 2
        elif lay_index == 0:
            self.clear()
        else:
            QMessageBox.about(self.ui,
                              '错误',
                              '没有可以删除的层！'
                              )
        self.display()

    def add_conv(self):
        global current_code, lay_index, network
        if lay_index == -1:
            QMessageBox.about(self.ui,
                              '错误',
                              '请先点击新建层!'
                              )
        else:
            in_channels = self.ui.in_channels.text()
            out_channels = self.ui.out_channels.text()
            kernel_size = self.ui.kernel_size.text()
            stride = self.ui.stride.text()
            padding = self.ui.padding.text()
            conv_code = '\t\t\tnn.Conv2d(in_channels = {}, out_channels = {}, kernel_size = {}, stride = {}, padding = {}, bias=False),\n'.format(
                in_channels, out_channels, kernel_size, stride, padding)
            linear_code = '\t\t\tnn.Linear(in_features = {}, out_features = {}, bias=True),\n'.format(in_channels,
                                                                                                      out_channels)
            if self.ui.comboBox.currentIndex() == 0:
                current_code = current_code
            elif self.ui.comboBox.currentIndex() == 1:
                current_code += conv_code
            elif self.ui.comboBox.currentIndex() == 2:
                current_code += linear_code
            network[lay_index] = current_code
            self.display()

    def add_activation(self):
        global current_code, network, lay_index
        if lay_index == -1:
            QMessageBox.about(self.ui,
                              '错误',
                              '请先点击新建层!'
                              )
        else:
            relu_code = '\t\t\tnn.ReLU(),\n'
            sigmoid_code = '\t\t\tnn.Sigmoid(),\n'
            softmax_code = '\t\t\tnn.Softmax(dim = 0),\n'
            if self.ui.comboBox_2.currentIndex() == 0:
                current_code = current_code
            elif self.ui.comboBox_2.currentIndex() == 1:
                current_code += sigmoid_code
            elif self.ui.comboBox_2.currentIndex() == 2:
                current_code += relu_code
            elif self.ui.comboBox_2.currentIndex() == 3:
                current_code += softmax_code
            network[lay_index] = current_code
            self.display()

    def add_nor(self):
        global current_code, network, lay_index
        if lay_index == -1:
            QMessageBox.about(self.ui,
                              '错误',
                              '请先点击新建层!'
                              )
        else:
            num_features = self.ui.nor_features_size.text()  # 获取批标准化输入通道数
            batch2d_code = '\t\t\tBatchNorm2d({0}),\n'.format(num_features)
            if self.ui.comboBox_3.currentIndex() == 0:
                current_code = current_code
            elif self.ui.comboBox_3.currentIndex() == 1:
                current_code += batch2d_code
            network[lay_index] = current_code
            self.display()

    def add_pool(self):
        global current_code, network, lay_index
        if lay_index == -1:
            QMessageBox.about(self.ui,
                              '错误',
                              '请先点击新建层!'
                              )
        else:
            kernel_size_x = self.ui.pool_kernel_size_x.text()
            kernel_size_y = self.ui.pool_kernel_size_y.text()
            stride_x = self.ui.pool_stride_x.text()
            stride_y = self.ui.pool_stride_y.text()
            max_code = '\t\t\tMaxPool2d(({0}, {1}), stride=({2}, {3})),\n'.format(kernel_size_x, kernel_size_y,
                                                                                  stride_x,
                                                                                  stride_y)
            avg_code = '\t\t\tAvgPool2d(({0}, {1}), stride=({2}, {3})),\n'.format(kernel_size_x, kernel_size_y,
                                                                                  stride_x,
                                                                                  stride_y)
            if self.ui.comboBox_4.currentIndex() == 0:
                current_code = current_code
            elif self.ui.comboBox_4.currentIndex() == 1:
                current_code += max_code
            elif self.ui.comboBox_4.currentIndex() == 2:
                current_code += avg_code
            network[lay_index] = current_code
            self.display()

    def add_dropout(self):
        global current_code, network, lay_index
        if lay_index == -1:
            QMessageBox.about(self.ui,
                              '错误',
                              '请先点击新建层!'
                              )
        else:
            drop_p = self.ui.drop_p.text()
            dropout_code = '\t\t\tDropout2d(p={0}),\n'.format(drop_p)
            current_code += dropout_code
            network[lay_index] = current_code
            self.display()

    def display(self):
        global network, lay_index, current_code
        code = ""
        for idx, li in enumerate(network):
            if idx != len(network) - 1:
                code = code + str(li) + "\n"
            else:
                code += str(li)
        self.ui.show_code.setText(code)  # 将list格式的网络代码转换成字符串输出
        # else:
        # self.ui.show_code.setText(current_code)

    def complete(self):
        global network, current_code, lay_index
        if lay_index == -1:
            QMessageBox.about(self.ui,
                              '错误',
                              '请先点击新建层!'
                              )
        else:
            len_code = len(current_code)
            if current_code[len_code - 1] == '\n':
                current_code += ')'
                network[lay_index] = current_code
            self.display()

    def clear(self):
        global lay_index, network, header_code, current_code
        lay_index = -1
        network = []
        current_code = '\t\tself.自定义网络名称 = nn.Sequential(\n'
        header_code = '\t\tself.自定义网络名称 = nn.Sequential(\n'
        self.display()

    def generate(self):
        global network, current_code, lay_index
        if lay_index == -1:
            QMessageBox.about(self.ui,
                              '错误',
                              '请先点击新建层!'
                              )
        else:
            len_code = len(current_code)
            if current_code[len_code - 1] == '\n':
                current_code += ')'
                network[lay_index] = current_code
            code = ""
            for idx, li in enumerate(network):
                if idx != len(network) - 1:
                    code = code + str(li) + "\n"
                else:
                    code += str(li)
            # 读取模板文件
            if self.ui.comboBox_5.currentIndex() == 0:
                with open('./MNIST/template.py', 'r', encoding='utf-8') as original_template:
                    template = original_template.read()
            elif self.ui.comboBox_5.currentIndex() == 1:
                with open('./Fashion_MNIST/template.py', 'r', encoding='utf-8') as original_template:
                    template = original_template.read()
            elif self.ui.comboBox_5.currentIndex() == 2:
                with open('./AE/template.py', 'r', encoding='utf-8') as original_template:
                    template = original_template.read()
            template = Template(template)
            # 模板替换
            result = template.substitute({'network': code})
            SaveFilePath = QFileDialog.getExistingDirectory(self.ui.generate, "请选择模板保存路径")  # 打开存储路径
            NewFilePath = SaveFilePath + '/Generated Network Files/'
            if not os.path.exists(NewFilePath):
                os.makedirs(NewFilePath)
            # 将替换后的代码写入到文件
            if self.ui.comboBox_5.currentIndex() == 0:
                with open(NewFilePath + 'MNIST_TRAIN.py', 'w', encoding='gbk') as generated_network:
                    generated_network.write(result)
            elif self.ui.comboBox_5.currentIndex() == 1:
                with open(NewFilePath + 'Fashion_MNIST_TRAIN.py', 'w', encoding='gbk') as generated_network:
                    generated_network.write(result)
            elif self.ui.comboBox_5.currentIndex() == 2:
                with open(NewFilePath + 'AENetwork.py', 'w', encoding='gbk') as generated_network:
                    generated_network.write(result)
            # fpath, fname = os.path.split('./cfg.py')  # 分离文件名和路径
            # shutil.copy('./cfg.py', NewFilePath + fname)  # 复制配套cfg配置文件
            os.startfile(NewFilePath)
