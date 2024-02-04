# !/usr/bin/env python
# -*-coding:utf-8 -*-
import os
import argparse
from model import DenseGAN
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
def check_args(args):
    check_folder(args.checkpoint_dir)
    check_folder(args.result_dir)
    check_folder(args.summary_dir)
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args
def parse_args():
    desc = "Tensorflow implementation of Fusion use GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--is_train', type=bool, default=True, help='True for training, False for testing [True]')
    # parser.add_argument('--is_train', type=bool, default=False, help='True for training, False for testing [True]')
    parser.add_argument('--sn', type=bool, default=True, help='using spectral norm')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_LoGall',help='Name of checkpoint directory [checkpoint_LoGallL')
    parser.add_argument('--result_dir', type=str, default='result', help='Directory name to save the generated images')
    parser.add_argument('--summary_dir', type=str, default='summary_dir',help='Name of summary_dir directory [summary]')
    # parser.add_argument('--log_dir', type=str, default='log_dir',help='Name of log_dir directory [log_dir]')
    parser.add_argument('--sample_dir', type=str, default='sample',help='Name of sample directory [sample]')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epoch [10]')
    parser.add_argument('--c_dim', type=int, default=1, help='Dimension of image color. [1]')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch images [128]')
    parser.add_argument('--image_size', type=int, default=122, help='The size of image to use [33]')
    # parser.add_argument('--image_size', type=int, default=132, help='he size of image to use [33]')
    parser.add_argument('--label_size', type=int, default=120, help='The size of image to use [33]')
    parser.add_argument('--stride', type=int, default=14, help='The size of stride to apply input image [14]')
    parser.add_argument('--learning_rate', type=float, default=1e-3,help='The learning rate of gradient descent algorithm [1e-4]')

    return check_args(parser.parse_args())

def main():
    args = parse_args()
    if args is None:
        exit()
    model = DenseGAN(args)
    model.train()

if __name__=='__main__':
    main()