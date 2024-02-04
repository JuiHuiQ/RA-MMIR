# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : iqa.py

Author     ：yujing_rao
"""
import cv2
import numpy as np
import math
import python
from python.brisquequality import *

def get_entropy(img_):
    x, y = img_.shape[0:2]
    # img_ = cv2.resize(img_, (100, 100))
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    img = np.array(img_)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res


def qem(ir_tensor_batch, vi_tensor_batch):
    b, c, h, w = ir_tensor_batch.shape
    qem_sum = 0
    for i in range(b):
        ir_temp = ir_tensor_batch[i, 0, ..., ...]
        vi_temp = vi_tensor_batch[i, 0, ..., ...]

        ir_np = ir_temp.cpu().numpy() * 127.5 + 127.5
        vi_np = vi_temp.cpu().numpy() * 127.5 + 127.5

        Q_ir = test_measure_BRISQUE1(ir_np)
        ir_np1 = ir_np.astype(int)
        res_ir = get_entropy(ir_np1)
        E1 = res_ir / 8

        Q_vi = test_measure_BRISQUE1(vi_np)
        vi_np1 = vi_np.astype(int)
        res_vi = get_entropy(vi_np1)
        E2 = res_vi / 8

        qem_tmp = (Q_vi * E2) / (Q_vi * E2 + Q_ir * E1)
        qem_sum += qem_tmp
    qem = qem_sum / b
    return qem



#
# dataset='INO'
# dataset_type='.bmp'
#
# ir_path='./INO/ir/'
# vi_path='./INO/vi/'
# ir_list=os.listdir(ir_path)
# dataset_len=len(ir_list)
# sem_sum=0
# for i in range(5):
#     ir_img_path=ir_path+str(i+1)+dataset_type
#     vi_img_path = vi_path + str(i+1) + dataset_type
#     Q1=test_measure_BRISQUE(ir_img_path)
#     image_ir = cv2.imread(ir_img_path, 0)
#     res_ir = get_entropy(image_ir)
#     E1=res_ir/8
#
#     Q2 = test_measure_BRISQUE(vi_img_path)
#     image_vi = cv2.imread(vi_img_path, 0)
#     res_vi = get_entropy(image_vi)
#     E2 = res_vi / 8
#
#     sem_tmp=(Q1*E2)/(Q1*E2+Q2*E1)
#     sem_sum+=sem_tmp
#     # print(sem_tmp)
# sem=sem_sum/dataset_len
#
# print(sem)