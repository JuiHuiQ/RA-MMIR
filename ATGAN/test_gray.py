import glob
import os
import time

import cv2
import numpy as np
import torch
# import xlwt

from .modules.generator import Generator

device = 'cuda:0'
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.png"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data

def input_setup(data_vi, data_ir, index):
    padding = 0
    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = imread_gray(data_ir[index])                                                   # 读取单通道图像
    _vi = imread_gray(data_vi[index])
    input_ir = (_ir - 127.5) / 127.5
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (_vi - 127.5) / 127.5
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi

def input_setup_all(data_vi, data_ir):
    padding = 0
    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = data_ir
    _vi = data_vi
    input_ir = (_ir - 127.5) / 127.5
    input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (_vi - 127.5) / 127.5
    input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
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

def imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)                                        # 读取单通道图像
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img[:, :]

def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def fusion_test(vi, ir, vi_color):
    g = Generator().to(device)

    weights = torch.load("L:\\CODE\\matching\\SuperGlue\\SuperGlue_training-main\\ATGAN\\checkpoint\\epoch_72\\model-72.pt")
    g.load_state_dict(weights)                                                                                                  # 载入权重
    g.eval()

    with torch.no_grad():
        train_data_ir, train_data_vi = input_setup_all(vi, ir)
        train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
        train_data_vi = train_data_vi.transpose([0, 3, 1, 2])

        train_data_ir = torch.tensor(train_data_ir).float().to(device)
        train_data_vi = torch.tensor(train_data_vi).float().to(device)
        start = time.time()
        result = g(train_data_ir, train_data_vi)  # 单通道融合图
        end = time.time()
        result = np.squeeze(result.cpu().numpy() * 127.5 + 127.5).astype(np.uint8)  # 单通道

        # save_o_path = os.path.join(path_o_g, str(i + 1) + ".jpg")
        # cv2.imwrite(save_o_path, result)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(result)  # 单通道

        # save_path = os.path.join(path_r, str(i + 1) + ".jpg")
        # save_c2g_path = os.path.join(path_r_c2g, str(i + 1) + ".jpg")
        t = end - start
        # sheet1.write(i, 0, t)  # 第0行第0列
        #
        img_vi = vi_color
        img_vi_rgb = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
        img_vi_ycrcb = cv2.cvtColor(img_vi_rgb, cv2.COLOR_RGB2YCR_CB)
        img_copy = img_vi_ycrcb
        img_copy[:, :, 0] = result
        img_f_bgr = cv2.cvtColor(img_copy, cv2.COLOR_YCR_CB2BGR)
        # cv2.imwrite(save_path, img_f_bgr)
        img_f_gray = cv2.cvtColor(img_f_bgr, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(save_c2g_path, img_f_gray)

    return img_f_bgr

# if __name__ == '__main__':
#     dataset_name=["INO", 'M3', 'MFNet', 'RoadScene', 'TNO']
#     for d in range(1, len(dataset_name)):
#         for e in range(72, 73):
#             print("test epoch" + str(e) + ' on the '+dataset_name[d]+'\n')
#             all(i=e, dataset=dataset_name[d])
