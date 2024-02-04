# encoding: utf-8

import torch
import random
from torchvision import datasets, transforms

def get_train_loader(args):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            'mnist_data',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.Lambda(lambda image: image.rotate(random.random() * args.angle * 2 - args.angle)),           # 空间变换
                transforms.ToTensor(),
            ]),
        ),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 0,                                        # 原为 4
        pin_memory = True if args.cuda else False,
    )

def get_test_loader(args):
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            'mnist_data',
            train = False,
            download = True,
            transform = transforms.Compose([
                transforms.Lambda(lambda image: image.rotate(random.random() * args.angle * 2 - args.angle)),           # 图像随机空间变换
                transforms.ToTensor(),
            ]),
        ),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 0,                                        # 原为 4
        pin_memory = True if args.cuda else False,
    )
