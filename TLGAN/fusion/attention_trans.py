# -*- coding: utf-8 -*-
# @Time    : 2022/1/6 9:45
# @Author  : Mina Han
# @FileName: attention_trans.py
# @Software: PyCharm
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from bn_lib.nn.modules import SynchronizedBatchNorm2d
import settings

import torch
import math
from torch import nn
norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class trans(nn.Module):
    def __init__(self):
        super(trans, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 3, kernel_size=1, stride=1))
        self.bn1 = nn.BatchNorm2d(3, momentum=0.9, eps=1e-5)

    def forward(self, ir_imgs):
        x = ir_imgs
        x = self.leaky_relu(self.bn1(self.sn_conv1(x)))

        return torch.tanh(x)


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
        self.v = k
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
        idn = x
        # The first 1x1 conv 第一个1 * 1卷积层
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)                                                 # b * c * n
        mu = self.mu.repeat(b, 1, 1)                                            # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)                                        # b * n * c
                z = torch.bmm(x_t, mu)                                          # b * n * k 求指数内积的指数，只是内积形式，核
                z = F.softmax(z, dim=2)                                         # b * n * k exp(z)，且归一化exp(z)
                # E步 求期望值
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))                    # 高斯混合求期望值
                # z_ = (self.v + 1) * (2. / self.v) / (1e-6 + self.v * (2. / self.v) - 2 * (2. / self.v) * torch.log(z)) # t分布求期望
                # M步 求参数更新
                mu = torch.bmm(x, z_)                                           # b * c * k 求出更新参数mu
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.
        # 移动平均操作写在train.py中，这很重要

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv 第二个1 * 1卷积层
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

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
