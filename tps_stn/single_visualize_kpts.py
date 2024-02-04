# encoding: utf-8
import time
import torch
from torch import nn
import itertools
import numpy as np
from PIL import Image
from .grid_sample import grid_sample
from torch.autograd import Variable
from .tps_grid_gen import TPSGridGen
import cv2

class TPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source_image, source_control_points, target_control_points, color = None):
        print(source_image.shape)
        source_image = np.array(source_image).astype('float32')
        source_image = np.expand_dims(source_image.swapaxes(2, 1).swapaxes(1, 0), 0)

        source_image = Variable(torch.from_numpy(source_image))

        _, _, source_height, source_width = source_image.size()
        # print(source_image.size())
        target_height = source_height
        target_width = source_width

        source_control_points_zero = np.zeros_like(source_control_points)
        target_control_points_zero = np.zeros_like(target_control_points)

        for i in range(0, source_control_points_zero.shape[0]):
            source_control_points_zero[i][0] = -1.0 + (source_control_points[i][0] / source_width) * 2.0
            source_control_points_zero[i][1] = -1.0 + (source_control_points[i][1] / source_height) * 2.0
            target_control_points_zero[i][0] = -1.0 + (target_control_points[i][0] / source_width) * 2.0
            target_control_points_zero[i][1] = -1.0 + (target_control_points[i][1] / source_height) * 2.0

        if target_control_points_zero.shape[0] > 5 :
            source_control_points_zero = torch.from_numpy(source_control_points_zero).to(torch.float32)                 # 源特征点-->torch

            target_control_points_zero = torch.from_numpy(target_control_points_zero).to(torch.float32)                 # 目标特征点-->torch

            tps = TPSGridGen(target_height, target_width, target_control_points_zero)                              # 建立TPSGridGen类对象tps, 输入目标点target

            source_coordinate, target_tps = tps(Variable(torch.unsqueeze(source_control_points_zero, 0)))          # 增加第0维度, 送入tps处理-->反向传播梯度; [1, H * W, 2] , ;

            # grid = source_coordinate.view(batch_size, target_height, target_width, 2)                                 # 分配到每个像素
            grid = source_coordinate.view(1, target_height, target_width, 2)                                            # 分配到每个像素
            # print(grid)

            if color == 1:
                canvas = Variable(torch.Tensor(1, 3, target_height, target_width).fill_(255))                     # 0--0, 梯度反向传播,构建空白“画布”
            else:
                canvas = Variable(torch.Tensor(1, 3, target_height, target_width).fill_(0))                 # 255--255, 梯度反向传播
            # print(canvas.size())

            target_image = grid_sample(source_image, grid, canvas)                                          # 对target_image进行填充

            target_image = target_image.data.numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)

            # target_image = Image.fromarray(target_image.astype('uint8'))

            # print(target_image.shape)

            # target_image.save('demo/target_avatar_1.jpg')

            return target_image
        else:

            source_image = source_image.numpy()
            source_image = source_image.squeeze(0)
            source_image = source_image.transpose(1, 2, 0)

            return source_image
