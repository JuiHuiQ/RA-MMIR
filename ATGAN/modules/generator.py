# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : generator.py
Create on  ：2022/11/9 16:46

Author     ：yujing_rao
"""
import math
import os

import cv2
import numpy as np
import torch
from torch import nn
import torch.utils.data
import torch
from .utils.conv import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        Ba1_inchannels = 4
        Ba1_outchannels = 1

        Ba2_inchannels = 8
        Ba2_outchannels = 2

        Se1_inchannels = 12
        se1_outchannels = 8

        Se2_inchannels = 12
        se2_outchannels = 8

        stride = 1
        self.leaky_relu = nn.LeakyReLU(0.2)
        # reflection_padding1 = int(np.floor(5 / 2))
        # self.reflection_pad1= nn.ReflectionPad2d(reflection_padding1)

        # ir
        self.ir_conv1 = Conv2d(1, 4, kernel_size=3, stride=1)
        self.ir_bn1 = nn.BatchNorm2d(4, momentum=0.9, eps=1e-5)

        self.ir_conv2 = Conv2d(5, 8, kernel_size=3, stride=1)
        self.ir_bn2 = nn.BatchNorm2d(8, momentum=0.9, eps=1e-5)

        self.ir_conv3 = Conv2d(13, 4, kernel_size=3, stride=1)
        self.ir_bn3 = nn.BatchNorm2d(4, momentum=0.9, eps=1e-5)

        # self.ir_conv4 = nn.utils.spectral_norm(nn.Conv2d(16, 8, kernel_size=3, stride=1))
        # self.ir_bn4 = nn.BatchNorm2d(8, momentum=0.9, eps=1e-5)

        # vi
        self.vi_conv1 = Conv2d(1, 4, kernel_size=3, stride=1)
        self.vi_bn1 = nn.BatchNorm2d(4, momentum=0.9, eps=1e-5)

        self.vi_conv2 = Conv2d(5, 8, kernel_size=3, stride=1)
        self.vi_bn2 = nn.BatchNorm2d(8, momentum=0.9, eps=1e-5)

        self.vi_conv3 = Conv2d(13, 4, kernel_size=3, stride=1)
        self.vi_bn3 = nn.BatchNorm2d(4, momentum=0.9, eps=1e-5)

        # self.vi_conv4 = nn.utils.spectral_norm(nn.Conv2d(16, 8, kernel_size=3, stride=1))
        # self.vi_bn4 = nn.BatchNorm2d(8, momentum=0.9, eps=1e-5)

        # self.brightA1=Bright_Attention(Ba1_inchannels,Ba1_outchannels)
        # self.brightA2=Bright_Attention(Ba2_inchannels,Ba2_outchannels)
        self.brightA1 = BrightM(Ba1_inchannels, Ba1_outchannels)
        self.brightA2 = BrightM(Ba2_inchannels, Ba2_outchannels)

        self.semantic1 = SemanticTrans(Se1_inchannels, se1_outchannels, kernel_size=3, stride=stride)
        self.semantic2 = SemanticTrans(Se2_inchannels, se2_outchannels, kernel_size=3, stride=stride)
        # self.semantic3 = SemanticTrans(Se3_inchannels, se3_outchannels,kernel_size=5,stride=stride)

        self.gfusion = GFusion(82,10)
        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)
    def forward(self,ir_img,vi_img):
        ir_layer1 = self.leaky_relu(self.ir_bn1(self.ir_conv1(ir_img)))


        ir_layer2 = self.leaky_relu(self.ir_bn2(self.ir_conv2(torch.cat([ir_img, ir_layer1], dim=1))))


        ir_layer3 = self.leaky_relu(self.ir_bn3(self.ir_conv3(torch.cat([ir_layer2, ir_img, ir_layer1], dim=1))))
        ir_layer31=torch.cat([ir_layer2,ir_layer3],dim=1)

        v_layer4, x_layer1,attention_layer4= self.brightA1(ir_layer3,ir_layer2)
        # 0 1 2
        v_layer3, x_layer2, attention_layer3 = self.brightA2(ir_layer2,ir_layer1)
        #34  56  78

        vi_layer1 = self.leaky_relu(self.vi_bn1(self.vi_conv1(vi_img)))


        vi_layer2 = self.leaky_relu(self.vi_bn2(self.vi_conv2(torch.cat([vi_img, vi_layer1], dim=1))))


        vi_layer3 = self.leaky_relu(self.vi_bn3(self.vi_conv3(torch.cat([vi_img, vi_layer1, vi_layer2], dim=1))))
        vi_layer31=torch.cat([vi_layer2,vi_layer3],dim=1)

        seman_layer2 = self.semantic1(vi_layer3, vi_layer2,vi_layer2)
        seman_layer1 = self.semantic2(seman_layer2, vi_layer1,vi_layer2)

        x_final = self.gfusion(ir_img, ir_layer1, torch.cat([ir_img, ir_layer1], dim=1),
                               ir_layer2, torch.cat([ir_layer2, ir_img, ir_layer1], dim=1),
                               ir_layer3, attention_layer4, v_layer4,
                          attention_layer3, v_layer3,vi_img, vi_layer1,
                               vi_layer2, vi_layer3, seman_layer2, seman_layer1)

        return torch.tanh(x_final)

class BrightM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BrightM, self).__init__()
        self.xconv = Conv2d(in_channels, out_channels, 1, 1, 0)
        self.qconv = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.kconv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.vconv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)
    def forward(self,x,x_up):
        x1=x_up
        x_out = self.xconv(x)
        q = self.qconv(x)
        q_batch, q_c, q_height, q_width = q.size()
        k = self.kconv(x)
        v = self.vconv(x)
        v_out = v
        q = q.view(q_batch, q_c, -1)
        k = k.view(q_batch, q_c, -1).permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        v_batch, v_c, v_height, v_width = v.size()
        v = v.view(v_batch, v_c, -1)
        # out = torch.bmm(attention, v)
        # out = out.view(v_batch, v_c, v_height, v_width)
        # out = self.gamma * out + x_out
        # out1=torch.cat([self.gamma * out,x_out],dim=1)
        # return out, v_out,out1
        out = torch.bmm(attention, v)
        out = out.view(v_batch, v_c, v_height, v_width)
        out1 = self.gamma * out
        out2 = out1 + x_out
        return v_out, x_out,out2

class SemanticTrans(nn.Module):
    def __init__(self,input_channels,out_channels,kernel_size,stride):
        super(SemanticTrans, self).__init__()
        self.sconv = Conv2d(input_channels, out_channels, kernel_size, stride)
        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)
    def forward(self,upper,current,second_layer):
        new_layer = torch.cat((upper, current), dim=1)
        Semanticlayer = self.sconv(new_layer)
        second_layer_semantic=second_layer+Semanticlayer
        return Semanticlayer

class GFusion(nn.Module):
    def __init__(self,input_channels,dense_channels):
        input_channel=input_channels+dense_channels
        super(GFusion, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1=Conv2d(input_channels+dense_channels, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(8, momentum=0.9, eps=1e-5)
        self.sn_conv2 = Conv2d(input_channels+dense_channels+8, 1, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1, momentum=0.9, eps=1e-5)
        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def forward(self, ir_img, ir_layer1, ir_layer11, ir_layer2, ir_layer21, ir_layer3, attention_layer4, v_layer4,
                          attention_layer3, v_layer3,vi_img, vi_layer1, vi_layer2, vi_layer3, seman_layer2, seman_layer1):

        ir_features=torch.cat([ir_img, ir_layer1, ir_layer11, ir_layer2, ir_layer21, ir_layer3, attention_layer4, v_layer4,
                          attention_layer3, v_layer3], dim=1)
        vi_features = torch.cat([vi_img, vi_layer1, torch.cat([vi_img, vi_layer1],dim=1), vi_layer2, torch.cat([vi_img, vi_layer1,vi_layer2],dim=1), vi_layer3, seman_layer2, seman_layer1],
                         dim=1)
        x=torch.cat([ir_features,vi_features],dim=1)
        x1 = self.leaky_relu(self.bn1(self.sn_conv1(x)))
        x2 = self.bn2(self.sn_conv2(torch.cat((x, x1), dim=1)))
        return x2


