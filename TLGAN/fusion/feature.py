# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 16:23
# @Author  : Mina Han
# @FileName: feature.py
# @Software: PyCharm
# !/usr/bin/env python
# -*-coding:utf-8 -*-
import torch

from torch import nn
# device ='cuda' if torch.cuda.is_available() else 'cpu'
class feature(nn.Module):
    def __init__(self):
        super(feature, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 256, kernel_size=3, stride=1))          # input_size=140  output_size=138
        self.bn1 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, stride=1))        # input_size=138   output_size=136
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 32, kernel_size=3, stride=1))         # input_size=136   output_size=134
        self.bn3 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 32, kernel_size=3, stride=1))          # input_size=134
        # self.bn4 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        # self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(32, 1, kernel_size=1, stride=1))

        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=7, stride=1))           # input_size=140 output_size=134
        self.bn5 = nn.BatchNorm2d(32, momentum=0.9, eps=1e-5)

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, imgs):
        x = imgs
        y = imgs
        # print(x.shape)
        # print(y.shape)
        x = self.leaky_relu(self.bn1(self.sn_conv1(x)))
        # print(x.shape)
        x = self.leaky_relu(self.bn2(self.sn_conv2(x)))
        # print(x.shape)
        x = self.leaky_relu(self.bn3(self.sn_conv3(x)))
        # print(x.shape)
        # x = self.leaky_relu(self.bn4(self.sn_conv4(x)))
        y = self.leaky_relu(self.bn5(self.sn_conv5(y)))
        # print(y.shape)
        x = torch.cat([x, y], dim=1)
        # print(x.shape)
        return torch.tanh(self.sn_conv4(x))