# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:56:34 2021

@author: 13479
"""
import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    ToTensor
)
import torchvision.models as models
from PIL import Image

# 显示可用GPU
print(f'Torch-Version {torch.__version__}')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')
print(torch.cuda.device_count())


# 构建模型
def model():
    model = models.resnet18(pretrained=False)
    # 全连接层的输入通道in_channels个数
    num_fc_in = model.fc.in_features
    # 改变全连接层，多分类问题，out_features = 5
    model.fc = nn.Linear(num_fc_in, 5)
    # 模型迁移到CPU/GPU (没有GPU可以删除这行)
    model = model.to(DEVICE)
    # 载入训练好的模型参数
    model.load_state_dict(torch.load("D:\\学习文件\\机器学习\\课程实践\\model.pth"))
    return model


# 模型实例化
model = model()


# 单张图片预测
def predict(model, imag_Path):
    # 疾病列表
    leaves_disease = ["木薯细菌性枯萎病", "木薯褐条病", "木薯绿色斑点", "木薯花叶病", "健康"]
    # 图像数据转换为torch.FloatTensor
    tensor = ToTensor()
    test_image_name = imag_Path
    custom_transform = Compose([
        tensor,
    ])
    # 打开图片
    test_image = Image.open(test_image_name)
    # 图片向量化
    test_image_tensor = custom_transform(test_image)
    # 将图片添加一个维度，符合模型输入
    test_image_tensor = torch.unsqueeze(test_image_tensor, dim=0)
    # 图片数据迁移到CPU/GPU (没有GPU可以删除这行)
    test_image_tensor = test_image_tensor.to(DEVICE)
    # 预测代码
    with torch.no_grad():
        # 模型转为预测模式，参数不会调整
        model.eval()
        pred = model(test_image_tensor)
        # 选取概率最大的值为预测值
        _, predicted = pred.max(1)
    return leaves_disease[predicted]


pre = predict(model, "./Picture/0/6103.jpg")
print(pre)
