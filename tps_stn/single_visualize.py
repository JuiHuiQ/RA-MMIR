# encoding: utf-8
import time
import torch
import itertools
import numpy as np
from PIL import Image
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen

source_image = Image.open('demo/source_avatar.jpg').convert(mode = 'RGB')

source_image = np.array(source_image).astype('float32')
print(source_image.shape)

source_image = np.expand_dims(source_image.swapaxes(2, 1).swapaxes(1, 0), 0)
print(source_image.shape)

source_image = Variable(torch.from_numpy(source_image))
_, _, source_height, source_width = source_image.size()
target_height = source_height
target_width = source_width

# creat control points; 创建控制点;
target_control_points = torch.Tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

# target_control_points = np.array([[445., 172.],
#  [410., 186.],
#  [426., 191.],
#  [436., 191.],
#  [386., 195.],
#  [398., 196.],
#  [423., 204.],
#  [352., 223.],
#  [422., 224.],
#  [448., 226.],
#  [384., 229.],
#  [421., 239.],
#  [356., 250.],
#  [345., 254.],
#  [334., 265.]])

# print(target_control_points.type)

source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1)

# source_control_points = np.array([[459., 418.],
#  [353., 498.],
#  [392., 498.],
#  [413., 496.],
#  [269., 525.],
#  [305., 523.],
#  [383., 524.],
#  [164., 602.],
#  [338., 565.],
#  [437., 558.],
#  [275., 583.],
#  [338., 596.],
#  [174., 659.],
#  [137., 672.],
#  [106., 709.]])

source_control_points = torch.from_numpy(source_control_points)
target_control_points = torch.from_numpy(target_control_points)

print('initialize module')
beg_time = time.time()
tps = TPSGridGen(target_height, target_width, target_control_points)
print(tps)
past_time = time.time() - beg_time
print('initialization takes %.02fs' % past_time)

source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
grid = source_coordinate.view(1, target_height, target_width, 2)
canvas = Variable(torch.Tensor(1, 3, target_height, target_width).fill_(255))
target_image = grid_sample(source_image, grid, canvas)
target_image = target_image.data.numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)
target_image = Image.fromarray(target_image.astype('uint8'))
target_image.save('demo/target_avatar_3.jpg')
