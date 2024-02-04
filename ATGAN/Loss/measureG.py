# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : measureG.py

Author     ：yujing_rao
"""
import torch
from torch import nn
from torch.nn import functional as F

class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
    def forward(self,I_f,I_s):
        I_sub=I_f-I_s
        # loss1= torch.square(torch.sqrt(torch.mean(torch.square(I_f- torch.mean(I_f)))) -torch.sqrt(torch.mean(torch.square(I_s- torch.mean(I_s)))))
        # loss2=torch.square(torch.mean(I_f)-torch.mean(I_s))
        loss3=2*torch.sqrt(torch.mean(torch.square(I_sub- torch.mean(I_sub))))
        loss=loss3
        return loss
class GLoss(nn.Module):
    def __init__(self,device):
        super(GLoss, self).__init__()
        self.ker_device=device
    def gradient(self,x):
        with torch.no_grad():
            laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
            kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(self.ker_device)
            return F.conv2d(x, kernel, stride=1, padding=1)
    def forward(self,I_1,I_2):
        loss=torch.mean(torch.square(self.gradient(I_1) - self.gradient(I_2)))
        return loss
