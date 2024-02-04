# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : wirte_h5.py

Author     ：yujing_rao
"""
import glob
import os

import cv2
import h5py
import numpy as np


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    return img[:, :]

def imread_gray(path):
    img = cv2.imread(path)/255
    return img[:, :,0]
def input_setup(config, data_dir, ir_flag,index=0):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    if config.is_train:
        # 取到所有的原始图片的地址
        data = prepare_data(config, dataset=data_dir)
    else:
        data = prepare_data(config, dataset=data_dir)

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6


    for i in range(len(data)):
        # input_, label_ = preprocess(data[i], config.scale)
        input_ = (imread(data[i]) - 127.5) / 127.5
        # input_ = imread(data[i]) / 255.
        # input_ = imread(data[i])/ 255
        label_ = input_

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        # 按14步长采样小patch
        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                sub_input = input_[int(x):int(x + config.image_size),
                            int(y):int(y + config.image_size)]  # [33 x 33]
                # 注意这里的padding，前向传播时由于卷积是没有padding的，所以实际上预测的是测试patch的中间部分
                sub_label = label_[int(x + padding):int(x + padding + config.label_size),
                            int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]
                # Make channel value
                if data_dir == "Train":
                    sub_input = cv2.resize(sub_input, (config.image_size / 4, config.image_size / 4),
                                           interpolation=cv2.INTER_CUBIC)
                    sub_input = sub_input.reshape([config.image_size / 4, config.image_size / 4, 1])
                    sub_label = cv2.resize(sub_label, (config.label_size / 4, config.label_size / 4),
                                           interpolation=cv2.INTER_CUBIC)
                    sub_label = sub_label.reshape([config.label_size / 4, config.label_size / 4, 1])
                    print('error')
                else:
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    # print(arrdata.shape)
    make_data(config, arrdata, arrlabel, data_dir)

    # if not config.is_train:
    #     print(nx, ny)
    #     print(h_real, w_real)
    #     return nx, ny, h_real, w_real


def make_data(config, data, label, data_dir):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if config.is_train:
        # savepath = os.path.join(os.getcwd(), os.path.join('checkpoint',data_dir,'train.h5'))
        savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'train.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    else:
        savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'test.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def prepare_data(config, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if config.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        # 将图片按序号排序
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    # print(data)

    return data


