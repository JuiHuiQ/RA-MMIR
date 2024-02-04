# encoding: utf-8

import torch
import itertools
import torch.nn as nn
from torch.autograd import Function, Variable

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    """计算部分流(偏爱矩阵)：输入点，控制点"""
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)                              # kpts - kpts_i
    # original implementation, very slow; 原来的实现，很慢;
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2)                                               # square of distance; 距离平方和;
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)                                           # U = r**2 * logr **2
    # fix numerical error for 0 * log(0), substitute all nan with 0; 修正0 * log(0)的数值错误，将所有nan替换为0;
    mask = repr_matrix != repr_matrix
    # print(mask)
    repr_matrix.masked_fill_(mask, 0)
    # print("0000", repr_matrix.size())
    return repr_matrix

class TPSGridGen(nn.Module):
    """网格tps"""
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)                                                                  # 目标控制点个数
        self.num_points = N
        target_control_points = target_control_points.float()                                              # 改变数据类型

        # create padded kernel matrix; 创建填充内核矩阵;
        forward_kernel = torch.zeros(N + 3, N + 3)                                                         # 创建0值torch
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)   # 计算径向基函数U(r)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)                                          # 复制矩阵
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

        # compute inverse matrix; 计算逆矩阵;
        inverse_kernel = torch.inverse(forward_kernel)                                                     # [N + 3, N + 3]
        # print(inverse_kernel.size())

        # create target cordinate matrix; 创建目标坐标(协调)矩阵;
        HW = target_height * target_width                                                                  # 平铺图像
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))             # 生成全局像素位置
        target_coordinate = torch.Tensor(target_coordinate)                                                # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1)                                                     # convert from (y, x) to (x, y); 转换;
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)    # 计算径向基函数U(r),点作用矩阵
        target_coordinate_repr = torch.cat([target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], dim = 1)

        # register precomputed matrices; 配准预计算的矩阵;
        self.register_buffer('inverse_kernel', inverse_kernel)                                             # 逆运算核:TPS函数
        self.register_buffer('padding_matrix', torch.zeros(3, 2))                                          # 池化矩阵大小:仿射变换尺寸
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        """输入源特征点，对源特诊点趋于目标特征点进行调整"""
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)                                                          # batch_size为特征点个数

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)   # 拼接torch:[1, N + 3, 2]
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)                                     # 张量相乘, 得到映射矩阵:[1, N + 3, 2]
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)             # 张量相乘，得到目标点坐标:[1, H * W, 2]

        return source_coordinate, mapping_matrix
