# encoding: utf-8

import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .grid_sample import grid_sample
from torch.autograd import Variable
from .tps_grid_gen import TPSGridGen

class CNN(nn.Module):
    """卷积神经网络"""
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class ClsNet(nn.Module):
    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x))

class BoundedGridLocNet(nn.Module):
    """受限梯度LocNet"""
    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))              # 反双曲正切
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):
    """不受限梯度LocNet, 目标控制点"""
    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)                                           # 返回特征点

class STNClsNet(nn.Module):
    def __init__(self):
        super(STNClsNet, self).__init__()
        # self.args = args
        """基本参数设定"""
        span_range_height = span_range_width = 0.9                                      # 采样步长
        grid_height = grid_width = grid_size = 16
        image_height = 480
        image_width = 640

        r1 = span_range_height
        r2 = span_range_width

        assert r1 < 1 and r2 < 1                                                                        # 断言
        # if >= 1, arctanh will cause error in BoundedGridLocNet; arctanh 将导致 BoundedGridLocNet 错误;

        # 梯度目标控制点，采样
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),)))                          # [-0.9  -0.3  0.3  0.9]

        Y, X = target_control_points.split(1, dim = 1)
        target_control_points = torch.cat([X, Y], dim = 1)

        # GridLocNet = {
        #     'unbounded_stn': UnBoundedGridLocNet,
        #     'bounded_stn': BoundedGridLocNet,
        # }[args.model]

        self.loc_net = UnBoundedGridLocNet(grid_height, grid_width, target_control_points)    # 3D点云定位网络

        self.tps = TPSGridGen(image_height, image_width, target_control_points)               # 网格tps

        self.cls_net = ClsNet()                                                                         # 卷积网络

    def forward(self, x):
        batch_size = x.size(0)                                                                          # 获取batch_size值

        source_control_points = self.loc_net(x)                                                         # 对图像x提取特征点

        source_coordinate = self.tps(source_control_points)                                             # 用tps对源特征点进行协调

        grid = source_coordinate.view(batch_size, self.image_height, self.image_width, 2)               # 网格

        transformed_x = grid_sample(x, grid)                                                            # 对x进行网格采样

        logit = self.cls_net(transformed_x)                                                             # 卷积网络

        return logit

def get_model():

    model = STNClsNet()

    return model
