# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 14:57
# @Author  : Mina Han
# @FileName: RGB2YCbcr.py
# @Software: PyCharm
import numpy as np
import cv2
import torch
import argparse
import torchvision.transforms
#####test##############
# def prepare_data(dataset):
#     data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
#     data = glob.glob(os.path.join(data_dir, "*.jpg"))
#     data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
#     data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
#     return data
##########test###########
# parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
# parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
# parser.add_argument('--batch_size', '-B', type=int, default=1)
# parser.add_argument('--gpu', '-G', type=int, default=-1)
# parser.add_argument('--num_workers', '-j', type=int, default=8)
# args = parser.parse_args()
def RGB2YCrCb(input_im):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def rgb2Y(rgb_img):
    ycbcr_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
    # print('ycbcr_img.shape=',ycbcr_img.shape)
    # print('ycbcr_img=',ycbcr_img)
    YImage = np.zeros((ycbcr_img.shape[0], ycbcr_img.shape[1]))
    CbImage = np.zeros((ycbcr_img.shape[0], ycbcr_img.shape[1]))
    CrImage = np.zeros((ycbcr_img.shape[0], ycbcr_img.shape[1]))
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            YImage[x][y] = ycbcr_img[x][y][0]
            CbImage[x][y] = ycbcr_img[x][y][1]
            CrImage[x][y] = ycbcr_img[x][y][2]
    # print('Y.shape=',YImage.shape)
    # print('Y=',YImage)
    return YImage, CbImage, CrImage
#
def Y2rgb(img,cb_img,cr_img):
    ycrcb_img = np.zeros((img.shape[0],img.shape[1],3))
    y_img = img
    # new_cr_img = cb_img
    # new_cb_img = cr_img
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            ycrcb_img[x][y][0] = y_img[x][y]
            ycrcb_img[x][y][1] = cb_img[x][y]
            ycrcb_img[x][y][2] = cr_img[x][y]
    ycrcb_img = ycrcb_img.astype(np.uint8)
    # print('ycrcb.shape=',ycrcb_img.shape)
    # print('ycrcb=',ycrcb_img)
    brg_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
    return brg_img


######test######
# if __name__ == '__main__':
#
#
#     img_1 = prepare_data(os.path.join('./img'))
#     img = cv2.imread(img_1[0])
#     y,cb,cr = rgb2Y(img)
#     rgb = Y2rgb(y,cb,cr)
######test######

