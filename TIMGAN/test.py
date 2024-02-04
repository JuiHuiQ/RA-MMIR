# !/usr/bin/env python
# -*-coding:utf-8 -*-

import imageio
import time
from utils import *
# from Generator import G
from .Gfusion import Gen
import numpy as np
import cv2
# # import kornia as K
# from thop import profile
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
def imsave(image, path):
    return imageio.imwrite(path, image)

def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    # data = glob.glob(os.path.join(data_dir, "*.png"))
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    # data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    # data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data

def input_setup(data_ir, data_vi):
    padding=1
    sub_ir_sequence = []
    sub_vi_sequence = []

    input_ir = (imread(data_ir) - 127.5) / 127.5

    input_ir=np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w2, h2=input_ir.shape
    input_ir=input_ir.reshape([w2, h2, 1])

    input_vi=(imread(data_vi)-127.5)/127.5

    input_vi=np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w4, h4=input_vi.shape
    input_vi=input_vi.reshape([w4, h4, 1])

    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)

    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)

    return train_data_ir, train_data_vi

def input_frame(vi, ir):
    padding = 0
    input_ir = (ir - 127.5) / 127.5
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (vi - 127.5) / 127.5
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    return input_vi, input_ir

def test_fuison(vi, ir):
    g = Gen().to(device)
    weights = torch.load('G:\\CODE\\matching\\SuperGlue\\SuperGlue_training-main\\ATGAN\\checkpoint\\epoch_72\\model-72.pt')
    g.load_state_dict(weights)
    g.eval()

    train_data_vi, train_data_ir = input_frame(vi, ir)
    print(train_data_ir.shape)
    train_data_ir = np.expand_dims(train_data_ir, axis=0)
    train_data_vi = np.expand_dims(train_data_vi, axis=0)

    train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
    train_data_vi = train_data_vi.transpose([0, 3, 1, 2])

    train_data_ir = torch.tensor(train_data_ir).float().to(device)
    train_data_vi = torch.tensor(train_data_vi).float().to(device)

    result = g(train_data_ir, train_data_vi)
    result = np.squeeze(result.cpu().numpy() * 127.5 + 127.5).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(result)

    return result


# if __name__ == '__main__':
#     # num_epoch=3
#     # for epoch in range(1,num_epoch):
#     #     test(str(epoch))
#     #     # print(epoch)
#     test(str(1))

