# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : calculate.py
Create on  ：2022/11/24 21:40

Author     ：yujing_rao
"""
import os

import torch
from thop import profile

from modules.generator import Generator
device = 'cpu'
g = Generator().to(device)
# if i<10:
#     weights = torch.load('checkpoint/epoch_fix'+str(i)+'/model-0'+str(i)+'.pt')
#     g.load_state_dict(weights)
# else:

weights = torch.load('checkpoint/epoch_721/model-72.pt')
g.load_state_dict(weights)
g.eval()
# model = build_detection_model(cfg)
print(g)
input = torch.randn(1, 1, 300, 300)
flop, para = profile(g, inputs=(input,input))
print('Flops:',"%.2fM" % (flop/1e6), 'Params:',"%.2fM" % (para/1e6))
total = sum([param.nelement() for param in g.parameters()])
print('Number of parameter: %.2fM' % (total/1e6))
