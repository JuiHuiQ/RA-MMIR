# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 19:06
# @Author  : Mina Han
# @FileName: RGB_run.py
# @Software: PyCharm

import time
from .Generator_ss import G
import torch
# from att import EMANet
# import settings
# from attention_trans import trans
import os
from .RGB2YCbcr import *
import glob
import imageio
import numpy as np
import cv2

# from zft_pp import globalEqualHist
# from zft_pp import localEqualHist
# import xlwt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.png"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data


##########multi-focus_image###########
def input_setup_y(data_vi, data_ir, index):
    padding = 4
    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = imread(data_ir[index])                                            # 一通道读取红外光图像
    _vi, img_cb, img_cr = imread_y(data_vi[index])                          # 可见光图像通道分离

    ######imshow########
    # cv2.imshow('y',_vi)
    # cv2.imshow('y',_ir)
    # cv2.imshow('cb',cb_img_vi)
    # cv2.imshow('cb',cb_img_ir)
    # cv2.imshow('cr',cr_img_vi)
    # cv2.imshow('cr',cr_img_ir)
    ######imshow--end########

    input_ir = (_ir - 127.5) / 127.5
    # input_ir = _ir / 255  ##归一化处理，这里没用
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (_vi - 127.5) / 127.5
    # input_vi = _vi / 255  ##归一化处理，这里没用
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi, img_cb, img_cr

def input_setup(vi, ir):
    padding = 4
    input_ir = (ir - 127.5) / 127.5
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (vi - 127.5) / 127.5
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])

    return input_vi, input_ir

   ######处理彩色图片############
def imread_y(path):
    img = cv2.imread(path)
    img_Y, img_cb, img_cr = rgb2Y(img)
    return img_Y, img_cb, img_cr

def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[:, :, 0]

def imsave(image, path):
    return imageio.imwrite(path, image)

def att_run(train_data_vi, train_data_ir):

    g = G().to(device)
    weights = torch.load('L:\\CODE\\matching\\SuperGlue\\SuperGlue_training-main\\TLGAN\\fusion\\checkpoint_1.5\\epoch1\\model-01.pth')            # 加载模型
    g.load_state_dict(weights)

    g.eval()

    with torch.no_grad():
        #####one######
        # train_data_ir, train_data_vi , cb_img_vi, cr_img_vi = input_setup_y(data_vi, data_ir, i)

        #####two######
        # print(train_data_ir.shape)
        train_data_vi, train_data_ir = input_setup(train_data_vi, train_data_ir)
        # print(train_data_vi.shape)
        train_data_ir = np.expand_dims(train_data_ir, axis=0)
        # train_data_ir = np.expand_dims(train_data_ir, axis=3)
        train_data_vi = np.expand_dims(train_data_vi, axis=0)
        # train_data_vi = np.expand_dims(train_data_vi, axis=3)

        train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
        train_data_vi = train_data_vi.transpose([0, 3, 1, 2])
        train_data_ir = torch.tensor(train_data_ir).float().to(device)
        train_data_vi = torch.tensor(train_data_vi).float().to(device)

        ####begin G######
        result_y = g(train_data_ir, train_data_vi)                                                                  # 运行主干网络

        result_y = np.squeeze(result_y.cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        # result_y = np.squeeze(result_y.cpu().numpy()).astype(np.uint8)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))                                                 # 用于生成自适应均衡化图像
        # result_y = clahe.apply(result_y)

        ###加入直方图均衡化后的结果####

        # result_y = globalEqualHist(result_y)
        # result_y = localEqualHist(result_y)

        ######cb和cr的数据就在cpu#######
        # cb_img_vi = cb_img_vi.astype(np.uint8)
        # cr_img_vi = cr_img_vi.astype(np.uint8)

        ######ycbcr2rgb######
        # result = Y2rgb(result_y, cb_img_vi, cr_img_vi)

        return result_y

# if __name__ == '__main__':
#     att_run()
