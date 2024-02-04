# -*-coding:utf-8-*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import settings
from bn_lib.nn.modules import SynchronizedBatchNorm2d
from torch.nn.modules.batchnorm import _BatchNorm

from functools import partial
from pathlib import Path

norm_layer = partial(SynchronizedBatchNorm2d, momentum = 3e-4)
# 网络层级

class VEMAU(nn.Module):
    """
    The Variation Expectation-Maximization Attention Unit (VEMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for VEM.
    """

    def __init__(self, c, k, stage_num = 2):
        super(VEMAU, self).__init__()
        self.stage_num = stage_num          # stage_num
        mu = torch.Tensor(1, c, k)          # 1 * c * k = 1 * 512 * 64
        mu.normal_(0, math.sqrt(2. / k))
        mu = self._l2norm(mu, dim=1)

        self.register_buffer('mu', mu)
        self.k = k                          # k = 64
        self.v = k - 1                      # v = k - 1
        self.theta = 45

        self.conv1 = nn.Conv2d(c, c, 1)
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
        # The first 1x1 conv
        x = self.conv1(x)

        # The VEM Attention.
        b, c, h, w = x.size()                # b, c, h, w
        x = x.view(b, c, h * w)              # b * c * n
        mu = self.mu.repeat(b, 1, 1)         # b * c * k

        theta = self.theta                   # 获取theta
        alpha_0 = float(math.pow(float(math.sin(theta * (math.pi / 180))), 2))
        beta_0 = float(math.pow(float(math.cos(theta * (math.pi / 180))), 2))

        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)
                z = torch.bmm(x_t, mu)
                '''V-E STEP'''
                z = alpha_0 * F.softmax(z, dim=2) + beta_0 * ((self.v + 1) * 1) / (self.v * 1 + 2 * 1 * z.norm(dim=2, keepdim=True))
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                '''V-M STEP'''
                mu = torch.bmm(x, z_)
                mu = self._l2norm(mu, dim=1)
        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)                                # b * k * n
        x = mu.matmul(z_t)                                      # b * c * n
        x = x.view(b, c, h, w)                                  # b * c * h * w
        x = F.relu(x, inplace=True)
        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu, theta

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim = dim, keepdim = True))

class CrossEntropyLoss2d(nn.Module):
    """交叉熵损失2d函数"""
    def __init__(self, weight = None, reduction = 'none', ignore_index = -1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)

class VGGBackbone_att(torch.nn.Module):
    def __init__(self, config, num_classes, input_channel = 1, device = 'gpu'):
        super(VGGBackbone_att, self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )   # block 3
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        vgg_path = Path(__file__).parent / 'superpoint_bn.pth'
        vgg_obj = torch.load(vgg_path, map_location=lambda storage, loc: storage.cuda())
        # print(vgg_obj.keys())
        vgg_obj = {k.replace('backbone.', ''): v for k, v in vgg_obj.items()}

        vgg_obj = {k: v for k, v in vgg_obj.items() if k in ['block1_1.0.weight', 'block1_1.0.bias',
                                                             'block1_2.0.weight', 'block1_2.0.bias',
                                                             'block2_1.0.weight', 'block2_1.0.bias',
                                                             'block2_2.0.weight', 'block2_2.0.bias',
                                                             'block3_1.0.weight', 'block3_1.0.bias',
                                                             'block3_2.0.weight', 'block3_2.0.bias',
                                                             'block4_1.0.weight', 'block4_1.0.bias',
                                                             'block4_2.0.weight', 'block4_2.0.bias']}

        self.load_state_dict(vgg_obj)
        self.emau = VEMAU(128, 32, settings.STAGE_NUM).cuda()
        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.crit = CrossEntropyLoss2d(ignore_index=255, reduction='none')


    def forward(self, x, lbl = None, size = None):
        """ 输入包括原始图像和ground-truth """
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)
        att, mu, theta = self.emau(feat_map)
        att_class = self.classifier(att)

        if size is None:
            size = x.size()[-2:]

        att_class = F.interpolate(att_class, size=size, mode='bilinear', align_corners=True)

        if self.training and lbl is not None:
            loss = self.crit(att_class, lbl)
            return loss, mu, theta
        else:
            return att, att_class

cconfig_1 = {
        'RAMM_Point': {
            'name': 'RAMM_Point',
            'using_bn': False,
            'grid_size': 8,
            'pretrained_model': 'none',
            'backbone': {
                'backbone_type': 'VGG',
                'vgg': {
                    'channels': [64, 64, 64, 64, 128, 128, 128, 128], }, },
            'det_head': {
                'feat_in_dim': 128},
            'des_head': {               # descriptor head
                'feat_in_dim': 128,
                'feat_out_dim': 256, },
            'det_thresh': 0.001,        # 1/65
            'nms': 4,
            'topk': -1,
        }
    }

def test_net():
    model = VGGBackbone_att(cconfig_1['RAMM_Point']['backbone']['vgg'],  21)
    model.eval()
    print(list(model.named_children()))

if __name__ == '__main__':
    test_net()