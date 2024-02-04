#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Time     :2022/7/119:32
@Author   :dan wu
@FileName :gen.py
@Software :PyCharm
"""

import torch
import cv2 as cv
from .utils import Gaussblur, Boxblur, Maxpool, MedianBlur, Laplacion, LoG, Sobel, Canny
from torch import nn
from torchvision import transforms
# from cabm import CBAM
# from Retinex import DecomNet, RelightNet, RelightNet1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# device ='cuda'
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sn_conv = nn.utils.spectral_norm(nn.Conv2d(1, 1, kernel_size=3, stride=1,bias=True))
        # self.gn = nn.GroupNorm(num_groups=2, num_channels=1)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 8, kernel_size=3, stride=1,bias=True))
        self.gn1 = nn.GroupNorm(num_groups=2,num_channels=8)


        # self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(16, 32, kernel_size=3, stride=1,bias=True))
        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(16, 16, kernel_size=3, stride=1,bias=True))
        self.gn2 = nn.GroupNorm(num_groups=2,num_channels=16)
        # self.gn2 = nn.GroupNorm(num_groups=2,num_channels=32)


        # self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1,bias=True))
        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(40, 32, kernel_size=3, stride=1,bias=True))
        self.gn3 = nn.GroupNorm(num_groups=2,num_channels=32)
        # self.gn3 = nn.GroupNorm(num_groups=2,num_channels=64)


        # self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1,bias=True))
        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(88, 64, kernel_size=3, stride=1,bias=True))
        self.gn4 = nn.GroupNorm(num_groups=2,num_channels=64)
        # self.gn4 = nn.GroupNorm(num_groups=2,num_channels=128)


        # self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(32, 16, kernel_size=3, stride=1,bias=True))
        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 32, kernel_size=3, stride=1,bias=True))
        # self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(64, 32, kernel_size=3, stride=1,bias=True))
        self.gn5 = nn.GroupNorm(num_groups=2, num_channels=32)
        # self.sn_conv6 = nn.utils.spectral_norm(nn.Conv2d(32, 8, kernel_size=3, stride=1,bias=True))
        self.sn_conv6 = nn.utils.spectral_norm(nn.Conv2d(32, 16, kernel_size=3, stride=1,bias=True))
        self.gn6 = nn.GroupNorm(num_groups=2, num_channels=16)
        self.sn_conv7 = nn.utils.spectral_norm(nn.Conv2d(16, 8, kernel_size=3, stride=1,bias=True))
        self.sn_conv8 = nn.utils.spectral_norm(nn.Conv2d(8, 1, kernel_size=1, stride=1,bias=True))

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self,ir_imgs,vi_imgs):

        ir_imgs = self.L(ir_imgs)
        vi_imgs = self.H(vi_imgs)
        v1l,v1h = self.decompose(self.leaky_relu(self.gn1(self.sn_conv1(vi_imgs))))                                     # 8
        i1l,i1h = self.decompose(self.leaky_relu(self.gn1(self.sn_conv1(ir_imgs))))
        x1v = self.H(torch.cat([v1h, i1h], dim=1))                                                                      # 16
        x1i = self.L(torch.cat([v1l, i1l], dim=1))
        v2l,v2h = self.decompose(self.pad(self.leaky_relu(self.gn2(self.sn_conv2(x1v)))))                               # 16
        i2l,i2h = self.decompose(self.pad(self.leaky_relu(self.gn2(self.sn_conv2(x1i)))))
        x2v = self.H(torch.cat([v2h, i2h,v1h], dim=1))                                                                  # 40
        x2i = self.L(torch.cat([v2l, i2l,i1l], dim=1))
        v3h,v3l = self.decompose(self.pad(self.leaky_relu(self.gn3(self.sn_conv3(x2v)))))                               # 32
        i3h,i3l = self.decompose(self.pad(self.leaky_relu(self.gn3(self.sn_conv3(x2i)))))
        x3v = self.H(torch.cat([v3h,i3h,v2h,v1h], dim=1))                                                               # 88
        x3i = self.L(torch.cat([v3l,i3l,i2l,i1l], dim=1))                                                               # 88
        v4 = self.pad(self.leaky_relu(self.gn4(self.sn_conv4(x3v))))                                                    # 64
        i4 = self.pad(self.leaky_relu(self.gn4(self.sn_conv4(x3i))))
        # x4 = v4+i4
        x4 = torch.cat([v4, i4], dim=1)                                                                                 # 128
        x5 = self.pad(self.leaky_relu(self.gn5(self.sn_conv5(x4))))                                                     # 32
        x6 = self.pad(self.leaky_relu(self.gn6(self.sn_conv6(x5))))                                                     # 16
        x7 = self.pad(self.leaky_relu(self.sn_conv7(x6)))                                                               # 8
        x8 = torch.tanh(self.sn_conv8(x7))





        return x8
    # def H(self,x):
    #     y = x+(x-Gaussblur(x))
        # y = x+(x-Boxblur(x))
        # y = x+(x-Maxpool(x))
        # y = x+(x-MedianBlur(x))
        # return y
    # def L(self,x):
    #     y = x+(Gaussblur(x))
        # y = x+(Boxblur(x))
        # y1 = Maxpool(x)
        # y = x+(y1)
        # y = x+(MedianBlur(x))
        # return y
    # def decompose(self,x):
    #     L = Gaussblur(x)
        # L = Boxblur(x)
        # L = Maxpool(x)
        # L = MedianBlur(x)
        # H = x - L
        # return L,H
    def H(self,x):
        # y = x + (Laplacion(x))
        y = x + (LoG(x))
        # y = x + (Sobel(x))
        # y = x + (Canny(x))
        return y
    def L(self,x):
        # y = x+(x-Laplacion(x))
        y = x-LoG(x)
        # y = x+(x-Sobel(x))
        # y = x+(x-Canny(x))
        return y
    def decompose(self,x):
        # H = Laplacion(x)
        H = LoG(x)
        # H = Sobel(x)
        # H = Canny(x)
        L = x - H
        return L,H