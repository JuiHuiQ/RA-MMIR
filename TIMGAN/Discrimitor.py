# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch
from torch import nn
# from  utils import IG,Ibo,Ima,Ime,IL,IS,IC,UG,Ubo,Uma,Ume,UL,US,UC
from utils import IR_test,Unsharp_mask

class Dv(nn.Module):
    def __init__(self):
        super(Dv, self).__init__()


        self.pad = nn.ReflectionPad2d(1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 16, kernel_size=3, stride=1))
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=16)

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(16, 32, kernel_size=3, stride=1))
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=32)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1))
        self.gn3 = nn.GroupNorm(num_groups=2, num_channels=64)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=3))
        self.gn4 = nn.GroupNorm(num_groups=2, num_channels=128)

        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=3))
        self.gn5 = nn.GroupNorm(num_groups=2, num_channels=256)

        self.linear = nn.Linear(11 * 11 * 256, 1)

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, x):
        # dst = thre(x)
        # x = dst * x

        x = Unsharp_mask(x)
        x1 = Unsharp_mask((self.leaky_relu(self.gn1(self.sn_conv1(x)))))  # 16
        x2 = Unsharp_mask((self.leaky_relu(self.gn2(self.sn_conv2(x1)))))  # 32
        x3 = Unsharp_mask((self.leaky_relu(self.gn3(self.sn_conv3(x2)))))  # 64
        x4 = Unsharp_mask(self.leaky_relu(self.gn4(self.sn_conv4(x3))))  # 128
        x5 = Unsharp_mask(self.leaky_relu(self.gn5(self.sn_conv5(x4))))
        x6 = x5.flatten(start_dim=1)
        x7 = self.linear(x6)



        return x7


class Di(nn.Module):
    def __init__(self):
        super(Di, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 16, kernel_size=3, stride=1))
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=16)

        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv2d(16, 32, kernel_size=3, stride=1))
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=32)

        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=1))
        self.gn3 = nn.GroupNorm(num_groups=2, num_channels=64)

        self.sn_conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=3))
        self.gn4 = nn.GroupNorm(num_groups=2, num_channels=128)

        self.sn_conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=3))
        self.gn5 = nn.GroupNorm(num_groups=2, num_channels=256)

        self.linear = nn.Linear(11 * 11 * 256, 1)

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, x):
        x = IR_test(x)
        x1 = IR_test((self.leaky_relu(self.gn1(self.sn_conv1(x)))))#16
        x2 = IR_test((self.leaky_relu(self.gn2(self.sn_conv2(x1)))))#32
        x3 = IR_test((self.leaky_relu(self.gn3(self.sn_conv3(x2)))))#64
        x4 = IR_test(self.leaky_relu(self.gn4(self.sn_conv4(x3))))#128
        x5 = IR_test(self.leaky_relu(self.gn5(self.sn_conv5(x4))))
        x6 = x5.flatten(start_dim=1)
        x7 = self.linear(x6)

        return x7
