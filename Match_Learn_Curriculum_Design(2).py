from PySide2 import QtCore
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFileDialog
from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QTableWidgetItem
from PySide2.QtGui import *
import torch
import os
from Model_Predict import model, predict
import PySide2.QtXml
import os

# 加上才能运行
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
# 加上才能显示图片
QtCore.QCoreApplication.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 前端界面
class Stats:
    num = 0
    method = "Simclr"  # 模型选择
    acc = "93%"  # 模型准确率
    picture = ""  # 图片路径
    lujing = ""  # 路径
    page = 0

    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('test(3).ui')
        self.ui.button.clicked.connect(self.openfile)
        self.ui.button_2.clicked.connect(self.Recognition)
        self.ui.up.clicked.connect(self.uppage)
        self.ui.down.clicked.connect(self.downpage)
        self.ui.combo.currentIndexChanged.connect(self.choice)
        str1 = "当前选择模型为： " + self.method + ",  准确率:  " + self.acc
        self.ui.statusbar.showMessage(str1)

    def openfile(self):  # 选择图片并显示
        FileDialog = QFileDialog(self.ui)
        FileDirectory = FileDialog.getOpenFileName(self.ui, "标题")  # 选择目录，返回选中的路径
        self.picture = os.path.abspath(FileDirectory[0])  # 获取图片的路径！！！！！！！！
        self.lujing = os.path.dirname(self.picture)
        print(self.picture)
        # print(self.lujing)
        picture = QPixmap(self.picture)
        self.ui.label.setPixmap(picture)

    def uppage(self):  # 上页功能
        # 计算当前文件所在位置
        self.page = 0
        num = 0
        for filename in os.listdir(self.lujing):
            if os.path.join(self.lujing, filename) == self.picture:
                break
            self.page += 1
        for filename in os.listdir(self.lujing):
            num += 1
            if num == self.page:
                self.picture = os.path.join(self.lujing, filename)
                break
        print(self.picture)
        picture = QPixmap(self.picture)
        self.ui.label.setPixmap(picture)

    def downpage(self):  # 下页功能
        # 计算当前文件所在位置
        self.page = 0
        num = -1
        for filename in os.listdir(self.lujing):
            if os.path.join(self.lujing, filename) == self.picture:
                break
            self.page += 1
        for filename in os.listdir(self.lujing):
            if num == self.page:
                self.picture = os.path.join(self.lujing, filename)
                break
            num += 1
        print(self.picture)
        picture = QPixmap(self.picture)
        self.ui.label.setPixmap(picture)

    def Recognition(self):  # 识别图片病症
        finish = predict(model, self.picture)
        color = "#ff0000"
        if finish == "健康":
            color = "#00ff00"
        self.ui.label_2.setText("<html><head/><body><p align='center'><span style=' font-size:16pt; "
                                "font-weight:600; "
                                "color:" + color + ";'>" + finish + "</span></p></body></html>")

        # 记录识别结果
        self.ui.table.insertRow(self.num)  # 插入第n行
        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignHCenter)  # 文本居中
        #
        item.setText(self.picture)
        self.ui.table.setItem(self.num, 0, item)

        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignHCenter)  # 文本居中
        item.setText(self.method)
        self.ui.table.setItem(self.num, 1, item)

        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignHCenter)  # 文本居中
        item.setText(finish)
        self.ui.table.setItem(self.num, 2, item)
        self.num += 1

    # 选择模型
    def choice(self):
        if self.ui.combo.currentText() == "Simclr":
            self.method = "Simclr"  # 模型选择
            self.acc = "93%"  # 模型准确率
        elif self.ui.combo.currentText() == "Resnet50":
            self.method = "Resnet50"  # 模型选择
            self.acc = "86%"  # 模型准确率
        elif self.ui.combo.currentText() == "VGG19":
            self.method = "VGG19"  # 模型选择
            self.acc = "90%"  # 模型准确率
        elif self.ui.combo.currentText() == "LeNet":
            self.method = "LeNet"  # 模型选择
            self.acc = "82%"  # 模型准确率
        elif self.ui.combo.currentText() == "DenseNet":
            self.method = "DenseNet"  # 模型选择
            self.acc = "86%"  # 模型准确率
        elif self.ui.combo.currentText() == "GoogleNet":
            self.method = "GoogleNet"  # 模型选择
            self.acc = "91%"  # 模型准确率
        str1 = "当前选择模型为： " + self.method + ",  准确率:  " + self.acc
        self.ui.statusbar.showMessage(str1)


app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()
