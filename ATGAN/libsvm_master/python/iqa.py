# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : iqa.py
Create on  ：2022/4/12 9:55

Author     ：yujing_rao
"""
import cv2
import numpy as np
import math
import python
from python.brisquequality import *
import sys
sys.path.append('./libsvm/python')
#求信息熵
def get_entropy(img_):
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (100, 100)) # 缩小的目的是加快计算速度
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

#求图像质量



# 对TNO数据集的训练集数据做评估

# 最后的SEM
dataset='INO'
dataset_type='.bmp'

ir_path='./INO/ir/'
vi_path='./INO/vi/'
ir_list=os.listdir(ir_path)
dataset_len=len(ir_list)
# Q1,E1计算ir; Q2,E2计算vi
sem_sum=0
for i in range(dataset_len):
    ir_img_path=ir_path+str(i+1)+dataset_type
    vi_img_path = vi_path + str(i+1) + dataset_type
    Q1=test_measure_BRISQUE(ir_img_path)
    image_ir = cv2.imread(ir_img_path, 0)
    res_ir = get_entropy(image_ir)
    E1=res_ir/8

    Q2 = test_measure_BRISQUE(vi_img_path)
    image_vi = cv2.imread(vi_img_path, 0)
    res_vi = get_entropy(image_vi)
    E2 = res_vi / 8

    sem_tmp=(Q1*E2)/(Q1*E2+Q2*E1)
    sem_sum+=sem_tmp
    # print(sem_tmp)
sem=sem_sum/dataset_len

print(sem)