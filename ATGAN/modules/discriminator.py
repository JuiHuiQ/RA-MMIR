# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : discriminator.py

Author     ：yujing_rao
"""
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
        # 38
        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=3, stride=2))
        # 17
        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)
        # 8
        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)
        # 3
        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2))
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)

        self.connected_layer1 = nn.Linear(in_features=7 * 7 * 256, out_features=64)
        self.connected_layer2 = nn.Linear(in_features=64, out_features=32)
        self.connected_layer3 = nn.Linear(in_features=32, out_features=2)
        self.softmax=nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        # self.leaky_relu = nn.LeakyReLU(0.2)
        # # 38
        # self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=3, stride=2))
        # # 17
        # self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        # self.bn1 = nn.BatchNorm2d(64, momentum=0.9, eps=1e-5)
        # # 8
        # self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2))
        # self.bn2 = nn.BatchNorm2d(128, momentum=0.9, eps=1e-5)
        # # 3
        # self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2))
        # self.bn3 = nn.BatchNorm2d(256, momentum=0.9, eps=1e-5)
        #
        # self.connected_layer1 = nn.Linear(in_features=7 * 7 * 256, out_features=64)
        # self.connected_layer2 = nn.Linear(in_features=64, out_features=32)
        # self.connected_layer3 = nn.Linear(in_features=32, out_features=2)
        # # self.softmax=nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

        # for name, p in self.named_parameters():
        #     if 'bias' in name:
        #         nn.init.zeros_(p)
        #     elif 'bn' in name:
        #         nn.init.trunc_normal_(p, mean=1, std=1e-3)
        #     else:
        #         nn.init.trunc_normal_(p, std=1e-3)
    def forward(self,x):
        x = self.leaky_relu(self.sn_conv1(x))
        x = self.leaky_relu(self.bn1(self.sn_conv2(x)))
        x = self.leaky_relu(self.bn2(self.sn_conv3(x)))
        x = self.leaky_relu(self.bn3(self.sn_conv4(x)))
        x_batch, x_c, x_height, x_width = x.size()
        # print(x_batch,x_c,x_height,x_width)
        x_ = x.contiguous().view(x_batch, -1)
        fc1 = self.connected_layer1(x_)
        fc2 = self.connected_layer2(fc1)
        fc3 = self.connected_layer3(fc2)

        return self.softmax(fc3)
