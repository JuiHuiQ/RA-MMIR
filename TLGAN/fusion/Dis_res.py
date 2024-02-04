# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 16:34
# @Author  : Mina Han
# @FileName: Dis_res.py
# @Software: PyCharm
# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
from torch import nn
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=3, stride=2))

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=2))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2))
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.sn_conv6 = nn.utils.spectral_norm(nn.Conv2d(1, 64, kernel_size=8, stride=4))
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)

        self.linear = nn.Linear(7 * 7 * 256, 1)


        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, x):
        # print(x.shape)
        y = self.leaky_relu(self.bn4(self.sn_conv6(x)))
        # print(y.shape)
        x = self.leaky_relu(self.sn_conv1(x))
        # print(x.shape)
        x = self.leaky_relu(self.bn1(self.sn_conv2(x)))
        # print(x.shape)
        x = torch.cat([x, y], dim=1)
        # print(x.shape)
        x = self.leaky_relu(self.bn2(self.sn_conv3(x)))
        # print(x.shape)
        x = self.leaky_relu(self.bn3(self.sn_conv4(x)))
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print(x.shape)
        return self.linear(x)