# encoding: utf-8

import os
import torch
import random
import argparse
import mnist_model
import data_loader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Training settings; 训练设置;
parser = argparse.ArgumentParser()                                              # 创建对象
parser.add_argument('--batch-size', type = int, default = 64)                   # batch_size 大小
parser.add_argument('--test-batch-size', type = int, default = 1000)            # test_batch_size 大小
parser.add_argument('--epochs', type = int, default = 10)                       # 训练步数
parser.add_argument('--lr', type = float, default = 0.01)                       # 学习率
parser.add_argument('--momentum', type=float, default = 0.5)                    # 冲量
parser.add_argument('--no-cuda', action = 'store_true', default = False)        # 是否使用cuda
parser.add_argument('--seed', type = int, default = 1)                          # 初始值
parser.add_argument('--log-interval', type = int, default = 10)                 # 对数间隔
parser.add_argument('--save-interval', type = int, default = 100)               # 保存间隔
parser.add_argument('--model', required = True)                                 # True==>含有STN的模型
parser.add_argument('--angle', type = int, default=60)                          # 倾角
parser.add_argument('--span_range', type = int, default = 0.9)                  # 跨度
parser.add_argument('--grid_size', type = int, default = 4)                     # 网格个数

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

"""定义基本参数"""
args.span_range_height = args.span_range_width = args.span_range
args.grid_height = args.grid_width = args.grid_size
args.image_height = args.image_width = 28

torch.manual_seed(args.seed)
"""使用cuda加载模型"""
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = mnist_model.get_model(args)                                                     # 加载带有tps的模型

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)       # 优化模型参数，随机梯度下降算法
train_loader = data_loader.get_train_loader(args)                                       # 加载train_loader(data_loader.get_train_loader)
test_loader = data_loader.get_test_loader(args)                                         # 加载test_loader(data_loader.get_test_loader)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)                                                                    # 模型处理后的输出

        loss = F.nll_loss(output, target)                                                       # 对数损失
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))                        # loss.data[0]

        if batch_idx % args.save_interval == 0:
            checkpoint_path = checkpoint_dir + 'epoch%03d_iter%03d.pth' % (epoch, batch_idx)
            torch.save(model.cpu().state_dict(), checkpoint_path)
            if args.cuda:
                model.cuda()

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile = True), Variable(target)
        output = model(data)                                                                    # data运行model
        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1]
        # get the index of the max log-probability; 得到最大对数概率的指数;
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)
    # loss function already averages over batch size; 损失函数已经在批大小上取平均值;
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.02f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy, ))
    log_file.write('{:.02f}\n'.format(accuracy))
    log_file.flush()
    os.fsync(log_file)

checkpoint_dir = 'checkpoint/%s_angle%d_grid%d/' % (args.model, args.angle, args.grid_size,)

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir('accuracy_log'):
    os.makedirs('accuracy_log')

log_file_path = 'accuracy_log/%s_angle%d_grid%d.txt' % (args.model, args.angle, args.grid_size,)

if __name__ == '__main__':
    with open(log_file_path, 'w') as log_file:
        for epoch in range(1, args.epochs + 1):             # arg.epochs = 10
            train(epoch)                                    # 训练
            test(epoch)                                     # 测试



