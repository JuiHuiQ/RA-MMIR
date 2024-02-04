# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 20:06
# @Author  : Mina Han
# @FileName: Generator_ss.py
# @Software: PyCharm

import torch
from functools import partial
import math

import numpy as np
import numpy
import matplotlib.pyplot as plt
# import cmap as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from bn_lib.nn.modules import SynchronizedBatchNorm2d
import settings
import math

from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
        期望最大化注意力单元(EMAU)。
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
        c (int):输入输出通道号。
        k (int):碱基数。
        stage_num (int): EM的迭代数。
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        self.v = k - 1
        self.theta = 45  # 定义初始混合系数参数

        mu = torch.Tensor(1, c, k)  # 1 * c * k
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm. 使用 Kaiming 范数初始化
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)  # c * c *1
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # print(x.shape)
        idn = x
        # The first 1x1 conv 第一个1 * 1卷积层
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k

        theta = self.theta  # 获取theta
        alpha_0 = float(math.pow(float(math.sin(theta * (math.pi / 180))), 2))          # 初始GMM混合系数
        beta_0 = float(math.pow(float(math.cos(theta * (math.pi / 180))), 2))           # 初始SMM混合系数

        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)                                                # b * n * c
                z = torch.bmm(x_t, mu)                                                  # b * n * k 求指数内积的指数，只是内积形式，核

                # E步 求期望值
                z = F.softmax(z, dim=2)                                                 # b * n * k exp(z)，且归一化exp(z)
                # z = alpha_0 * F.softmax(z, dim=2) + beta_0 * ((self.v + 1) * 1) / (self.v * 1 + 2 * 1 * z.norm(dim=2, keepdim=True))
                # z = F.softmax(z, dim=2) + ((self.v + 1) * 1) / (self.v * 1 + 2 * 1 * z.norm(dim=2, keepdim=True))
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))                            # 高斯混合求期望值

                # M步 求参数更新
                mu = torch.bmm(x, z_)                                                   # b * c * k 求出更新参数mu
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant;
        # 移动平均操作写在train.py中，这很重要;

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w

        x = F.relu(x, inplace=True)

        # The second 1x1 conv 第二个1 * 1卷积层
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)
        # x = torch.from_numpy(x)
        # print(x.shape)

        # one_pmap = x[0, 0, :, :] * 255  # 61x61 可视化后的确能表达某种分布特征
        '''可视化前景概率图（P的前一半）'''
        # plt.imshow(numpy.array(one_pmap.cpu()))
        # plt.show()
        '''三维可视化前景概率图'''
        # my_col = cm.jet(np.random.rand(one_pmap.shape[0], one_pmap.shape[1]))
        # x_1 = y = numpy.arange(start=0, stop=61)
        # X, Y = numpy.meshgrid(x_1, y)
        # fig = plt.figure(figsize=(12, 10))
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(X, Y, numpy.array(one_pmap.cpu()), cmap=cm.jet)  # facecolors=my_col,
        # ax.view_init(60, -30)
        # plt.show()
        return x

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class G(nn.Module):
    """生成对抗主干网络"""
    def __init__(self):
        super(G, self).__init__()
        self.sn_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.sn_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.sn_conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.sn_conv4 = nn.Conv2d(1, 32, kernel_size=7, stride=1)

        self.emau = EMAU(32, 64, 3)                                                                 # 载入EMAU

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pad = nn.ReflectionPad2d(1)

        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1))
        self.bn1 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.sn_conv6 = nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, stride=1))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv7 = nn.utils.spectral_norm(nn.Conv2d(384, 32, kernel_size=3, stride=1))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        self.sn_conv8 = nn.utils.spectral_norm(nn.Conv2d(32, 1, kernel_size=1, stride=1))

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, ir_imgs, vi_imgs):
        x_v = self.sn_conv1(vi_imgs)
        y_v = self.sn_conv4(vi_imgs)
        x_v = self.sn_conv2(x_v)
        x_v = self.sn_conv3(x_v)
        x_v = self.emau(x_v)
        # print(y_v.type)
        # print(x_v.type)
        v = torch.cat([x_v, y_v], dim=1)
        # print(v.shape)
        x_r = self.sn_conv1(ir_imgs)
        y_r = self.sn_conv4(ir_imgs)
        x_r = self.sn_conv2(x_r)
        x_r = self.sn_conv3(x_r)
        x_r = self.emau(x_r)
        r = torch.cat([x_r, y_r], dim=1)
        # print(r.shape)

        x_n = torch.cat([v, r], dim=1)
        x_1 = self.sn_conv5(x_n)
        x_1 = self.pad(x_1)
        x_1 = torch.cat([x_1, x_n], dim=1)
        x_2 = self.sn_conv6(x_1)
        x_2 = self.pad(x_2)
        x_2 = torch.cat([x_2, x_1], dim=1)
        x_3 = self.sn_conv7(x_2)
        # x_3 = torch.cat([x_3,x_n])
        x = self.sn_conv8(x_3)

        return torch.tanh(x)