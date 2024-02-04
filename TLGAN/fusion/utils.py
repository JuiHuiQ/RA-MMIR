# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : utils.py
Create on  ：2021/7/26 14:07

Author     ：yujing_rao
"""
import imageio
import glob
import os
import h5py
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2
# from RGB2YCbcr import *

device ='cuda' if torch.cuda.is_available() else 'cpu'


def gradient(x):
    with torch.no_grad():
        laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
        return F.conv2d(x, kernel, stride=1, padding=1)
def gradient_L(x):
    with torch.no_grad():
        laplace = [[0.0, 0.1, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
        return F.conv2d(x, kernel, stride=1, padding=1)


# def read_data(path):
#     """
#     Read h5 format data file
#
#     Args:
#       path: file path of desired file
#       data: '.h5' file format that contains train data values
#       label: '.h5' file format that contains train label values
#     """
#     with h5py.File(path, 'r') as hf:
#         data = np.array(hf.get('data'))
#         label = np.array(hf.get('label'))
#         return data, label

# def preprocess(path, scale=3):
#   """
#   Preprocess single image file
#     (1) Read original image as YCbCr format (and grayscale as default)
#     (2) Normalize
#     (3) Apply image file with bicubic interpolation
#
#   Args:
#     path: file path of desired file
#     input_: image applied bicubic interpolation (low-resolution)
#     label_: image with original resolution (high-resolution)
#   """
#   #读到图片
#   image = imread(path, is_grayscale=True)
#   #将图片label裁剪为scale的倍数
#   label_ = modcrop(image, scale)
#
#   # Must be normalized
#   image = (image-127.5 )/ 127.5
#   label_ = (image-127.5 )/ 127.5
#   #下采样之后再插值
#   input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
#   input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
#
#   return input_, label_

def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[:, :, 0]

# def imread_Y(path):
#     img = cv2.imread(path)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     img,img_cb,img_cr = rgb2Y(img)
#     return img


def prepare_data(config, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if config.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))
        data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        # 将图片按序号排序
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(data_dir, "*.png"))
        data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    print(data)

    return data

# def prepare_data(dataset):
#     """
#     Args:
#       dataset: choose train dataset or test dataset
#
#       For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
#     """
#     # if config.is_train:
#     #     filenames = os.listdir(dataset)
#     #     data_dir = os.path.join(os.getcwd(), dataset)
#     #     data = glob.glob(os.path.join(data_dir, "*.bmp"))
#     #     data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
#     #     # 将图片按序号排序
#     #     data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
#     # else:
#     data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
#     data = glob.glob(os.path.join(data_dir, "*.bmp"))
#     data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
#     data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
#     # print(data)
#
#     return data

def make_data(config, data, label,data_dir):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if config.is_train:
    #savepath = os.path.join(os.getcwd(), os.path.join('checkpoint',data_dir,'train.h5'))
    savepath = os.path.join('.', os.path.join('checkpoint',data_dir,'train.h5'))
    if not os.path.exists(os.path.join('.',os.path.join('checkpoint',data_dir))):
        os.makedirs(os.path.join('.',os.path.join('checkpoint',data_dir)))
  else:
    savepath = os.path.join('.', os.path.join('checkpoint',data_dir,'test.h5'))
    if not os.path.exists(os.path.join('.',os.path.join('checkpoint',data_dir))):
        os.makedirs(os.path.join('.',os.path.join('checkpoint',data_dir)))
  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def input_setup(config, data_dir, index=0):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    if config.is_train:
        # 取到所有的原始图片的地址
        data = prepare_data(config,dataset=data_dir)
    else:
        data = prepare_data(config,dataset=data_dir)

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6  #自动计算padding

    if config.is_train:
        for i in range(len(data)):
            # input_, label_ = preprocess(data[i], config.scale)
            #####灰度图读取######
            input_ = (imread(data[i]) - 127.5) / 127.5
            #####读取Y通道#######
            # input_ = (imread_Y(data[i]) - 127.5) / 127.5

            label_ = input_

            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            # 按14步长采样小patch
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[int(x):int(x + config.image_size), int(y):int(y + config.image_size)]  # [33 x 33]
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

    else:
        # input_, label_ = preprocess(data[2], config.scale)
        # input_=np.lib.pad((imread(data[index])-127.5)/127.5,((padding,padding),(padding,padding)),'edge')
        # label_=input_
        input_ = (imread(data[index]) - 127.5) / 127.5
        if len(input_.shape) == 3:
            h_real, w_real, _ = input_.shape
        else:
            h_real, w_real = input_.shape
        padding_h = config.image_size - ((h_real + padding) % config.label_size)
        padding_w = config.image_size - ((w_real + padding) % config.label_size)
        input_ = np.lib.pad(input_, ((padding, padding_h), (padding, padding_w)), 'edge')
        label_ = input_
        h, w = input_.shape
        # print(input_.shape)
        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                sub_label = label_[x + padding:x + padding + config.label_size,
                            y + padding:y + padding + config.label_size]  # [21 x 21]

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
    make_data(config,arrdata, arrlabel, data_dir)

    if not config.is_train:
        print(nx, ny)
        print(h_real, w_real)
        return nx, ny, h_real, w_real

def input_setup_mask(config, data_dir, index=0):
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
    # padding = abs(config.image_size - config.label_size) / 2  # 6  #自动计算padding

    if config.is_train:
        for i in range(len(data)):
            # input_, label_ = preprocess(data[i], config.scale)
            #####灰度图读取######
            input_ = (imread(data[i]) - 127.5) / 127.5
            #####读取Y通道#######
            # input_ = (imread_Y(data[i]) - 127.5) / 127.5

            label_ = input_

            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            # 按14步长采样小patch
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[int(x):int(x + config.image_size), int(y):int(y + config.image_size)]  # [33 x 33]
                    # 注意这里的padding，前向传播时由于卷积是没有padding的，所以实际上预测的是测试patch的中间部分
                    # sub_label = label_[int(x + padding):int(x + padding + config.label_size),
                    #             int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]
                    # Make channel value
                    if data_dir == "Train":
                        sub_input = cv2.resize(sub_input, (config.image_size / 4, config.image_size / 4),
                                               interpolation=cv2.INTER_CUBIC)
                        sub_input = sub_input.reshape([config.image_size / 4, config.image_size / 4, 1])
                        # sub_label = cv2.resize(sub_label, (config.label_size / 4, config.label_size / 4),
                        #                        interpolation=cv2.INTER_CUBIC)
                        # sub_label = sub_label.reshape([config.label_size / 4, config.label_size / 4, 1])
                        print('error')
                    else:
                        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                        # sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    # sub_label_sequence.append(sub_label)

    else:
        # input_, label_ = preprocess(data[2], config.scale)
        # input_=np.lib.pad((imread(data[index])-127.5)/127.5,((padding,padding),(padding,padding)),'edge')
        # label_=input_
        input_ = (imread(data[index]) - 127.5) / 127.5
        if len(input_.shape) == 3:
            h_real, w_real, _ = input_.shape
        else:
            h_real, w_real = input_.shape
        # padding_h = config.image_size - ((h_real + padding) % config.label_size)
        # padding_w = config.image_size - ((w_real + padding) % config.label_size)
        # input_ = np.lib.pad(input_, ((padding, padding_h), (padding, padding_w)), 'edge')
        label_ = input_
        h, w = input_.shape
        # print(input_.shape)
        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1;
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                # sub_label = label_[x + padding:x + padding + config.label_size,
                #             y + padding:y + padding + config.label_size]  # [21 x 21]

                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                # sub_label_sequence.append(sub_label)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    # arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    # print(arrdata.shape)
    make_data(config, arrdata, data_dir)

    if not config.is_train:
        print(nx, ny)
        print(h_real, w_real)
        return nx, ny, h_real, w_real


def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label')).astype('float32')
        return data, label
def read_data_mask(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        # label = np.array(hf.get('label'))
        return data

def get_images(data_dir,image_size,label_size,stride,is_train):
    data=prepare_data(data_dir)
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) // 2  # 6
    if is_train:
        for i in range(len(data)):
            input_ = imread(data[i] - 127.5) / 127.5
            label_ = input_
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            for x in range(0, h - image_size + 1, stride):
                for y in range(0, w - image_size + 1, stride):
                    sub_input = input_[x:x + image_size, y:y + image_size]  # [33 x 33]
                    # 注意这里的padding，前向传播时由于卷积是没有padding的，所以实际上预测的是测试patch的中间部分
                    sub_label = label_[x + padding:x + padding + label_size,
                                y + padding:y + padding + label_size]  # [21 x 21]
                    # Make channel value

                    sub_input = sub_input.reshape([image_size, image_size, 1])
                    sub_label = sub_label.reshape([label_size, label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)
    sub_input_sequence = np.asarray(sub_input_sequence, dtype=np.float32)
    sub_label_sequence = np.asarray(sub_label_sequence, dtype=np.float32)
    return sub_input_sequence, sub_label_sequence

def imsave(image, path):
    return imageio.imwrite(path, image)

