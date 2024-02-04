# encoding: utf-8
import os
import glob
import torch
import random
import argparse
from .mnist_model import get_model
# import data_loader
# import numpy as np
# from .mnist_model import STNClsNet
from .grid_sample import grid_sample
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
# from torchvision import datasets, transforms

"""可视化"""

# span_range_height = span_range_width = span_range = 0.9
# grid_height = grid_width = grid_size = 16
# image_height = image_width = 28

grid_height = grid_width = grid_size = 16

# args.cuda = torch.cuda.is_available()
# random.seed(1024)

# assert args.model in ['bounded_stn', 'unbounded_stn']
model = get_model().cuda()

# image_dir = 'image/%s_angle%d_grid%d/' % (args.model, args.angle, args.grid_size)                                       # 图片路径

# if not os.path.isdir(image_dir):
#     os.makedirs(image_dir)

# test_loader = data_loader.get_test_loader(args)                                                                         # 获取图像
# target2data_list = {i: [] for i in range(10)}
# total = 0
# N = 10

# for data_batch, target_batch in test_loader:                                                                            # 引入测试数据
#     for data, target in zip(data_batch, target_batch):
#         data_list = target2data_list[int(target)]
#         if len(data_list) < N:
#             data_list.append(data)
#             total += 1
#     if total == N * 10:
#         break
#
# data_list = [target2data_list[i][j] for i in range(10) for j in range(N)]
# source_data = torch.stack(data_list).cuda()                                                                                # 堆载入数据

# batch_size = N * 10
# frames_list = [[] for _ in range(batch_size)]

# paths = sorted(glob.glob('checkpoint/%s_angle%d_grid%d/*.pth' % (args.model, args.angle, args.grid_size, )))[::-1]

font = ImageFont.truetype('Comic Sans MS.ttf', 20)


def tps_stn(image, source, target):
    path = "I:\\PRAI\\CODE\\CODE\\STN\\tps_stn_pytorch-master\\checkpoint\\unbounded_stn_angle90_grid8\\epoch010_iter900.pth"         # 权重路径
    model.load_state_dict(torch.load(path))                                         # 载入权重
    source_control_points = model.loc_net(Variable(source, volatile = True))   # 提取源点
    source_coordinate = model.tps(source_control_points)                            # 对源点进行tps变换
    grid = source_coordinate.view(batch_size, 28, 28, 2)                            # 网格
    target_data = grid_sample(source, grid).data                               # 目标数据

    source_array = (source[:, 0] * 255).cpu().numpy().astype('uint8')          # 源特征点数组
    target_array = (target_data[:, 0] * 255).cpu().numpy().astype('uint8')          # 目标特征点数组

    # for si in range(batch_size):                                                    # sample index; 样本指标;
    # resize for better visualization; 调整大小以更好地可视化;
    source_image = Image.fromarray(source_array).convert('RGB').resize((128, 128))      # 对源图像进行处理
    target_image = Image.fromarray(target_array).convert('RGB').resize((128, 128))      # 对目标图像进行处理

    # create grey canvas for external control points; 为外部控制点创建灰色画布; 进行采样重构;
    canvas = Image.new(mode = 'RGB', size = (64 * 7, 64 * 4), color = (128, 128, 128))      # 构建图像
    canvas.paste(source_image, (64, 64))                                                    # 模糊
    canvas.paste(target_image, (64 * 4, 64))
    source_points = source_control_points.data
    source_points = (source_points + 1) / 2 * 128 + 64
    draw = ImageDraw.Draw(canvas)

    for x, y in source_points:
        draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill = (255, 0, 0))

    source_points = source_points.view(grid_size, grid_size, 2)                   # 重构源点

    for j in range(grid_size):
        for k in range(grid_size):
            x1, y1 = source_points[j, k]
            if j > 0:                                                       # connect to left; 连接到左边;
                x2, y2 = source_points[j - 1, k]
                draw.line((x1, y1, x2, y2), fill = (255, 0, 0))
            if k > 0:                                                       # connect to up; 搭接;
                x2, y2 = source_points[j, k - 1]
                draw.line((x1, y1, x2, y2), fill = (255, 0, 0))

        # draw.text((10, 0), 'sample %03d, iter %03d' % (si, len(paths) - 1 - pi), fill = (255, 0, 0), font = font)
        # canvas.save(image_dir + 'sample%03d_iter%03d.png' % (si, len(paths) - 1 - pi))                                  # 保存

    return 0
