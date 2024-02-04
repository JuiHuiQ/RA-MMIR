# !/usr/bin/env python
# -*-coding:utf-8 -*-

import glob
import os
# import h5py
import numpy as np
from math import exp
import torch
from torch.autograd import Variable
from torch.nn import functional as F
# from basicsr.archs.vgg_arch import VGGFeatureExtractor
import cv2
import torch.nn as nn
# import kornia as k


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# device = 'cpu'
# device ='cuda'
def gradient_L(x):
    with torch.no_grad():
        # laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        laplace = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
        return F.conv2d(x, kernel, stride=1, padding=1)


def gradient_Sh(x):
    # Scharr函数
    filter_x = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]
    kernel_x = torch.FloatTensor(filter_x).unsqueeze(0).unsqueeze(0).to(device)
    filter_y = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]]
    kernel_y = torch.FloatTensor(filter_y).unsqueeze(0).unsqueeze(0).to(device)
    d_x = F.conv2d(x, kernel_x, stride=1, padding=1)
    d_y = F.conv2d(x, kernel_y, stride=1, padding=1)
    d = torch.sqrt(torch.square(d_x) + torch.square(d_y))
    return d


def gradient_So(x):  # 求梯度信息(纹理？)
    # Sobel算子
    filter_x = [[-1.0, 0.0, 1.0], [-12.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    kernel_x = torch.FloatTensor(filter_x).unsqueeze(0).unsqueeze(0).to(device)
    filter_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    kernel_y = torch.FloatTensor(filter_y).unsqueeze(0).unsqueeze(0).to(device)
    d_x = F.conv2d(x, kernel_x, stride=1, padding=1)
    d_y = F.conv2d(x, kernel_y, stride=1, padding=1)
    d = torch.sqrt(torch.square(d_x) + torch.square(d_y))

    # print('d:',d)
    return d

def Fro_LOSS(batchimg):
    fro_norm = torch.square(torch.norm(batchimg, p='fro')) / (int(batchimg.shape[1]) * int(batchimg.shape[2]))
    E = torch.mean(fro_norm)
    # print('E:',E)
    return E

def grad(img):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(device)
    g = F.conv2d(img, kernel, stride = 1, padding = 1)
    return g


def Gaussblur(input):
    # 高斯模糊是低通滤波的一种， 也就是滤波函数是低通高斯函数
    m = nn.ReflectionPad2d(1)
    x = k.filters.gaussian_blur2d(input, (13, 13), (1.5, 1.5))
    return x
def Boxblur(input):
    x = k.filters.box_blur(input,(3, 3))
    return x
def Maxpool(input):
    m = nn.ReflectionPad2d(1)
    x = k.filters.max_blur_pool2d(input,kernel_size=3)
    return m (x)
def MedianBlur(input):
    x = k.filters.median_blur(input,(3,3))
    return x

def Canny(input):
    x = k.filters.canny(input)[1]
    return x
def Laplacion(input):
    x = k.filters.laplacian(input, kernel_size=3)
    return x
def Sobel(input):
    x = k.filters.sobel(input)
    return x
def LoG(input):
    x_gau=k.filters.gaussian_blur2d(input, (13, 13), (1, 1))
    x_gau_la=k.filters.laplacian(x_gau, kernel_size=3)
    return x_gau_la

def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min)






def IR_test(x):
    I = IG(x)
    # I = Ibo(x)
    # I = Ima(x)
    # I = Ime(x)
    # I = IL(x)
    # I = IS(x)
    # I = IC(x)
    # I = ILoG(x)
    return I
def Unsharp_mask(x):
    U = UG(x)
    # U = Ubo(x)
    # U = Uma(x)
    # U = Ume(x)
    # U = UL(x)
    # U = US(x)
    # U = UC(x)
    # U = ULoG(x)
    return U



def UG(input):
    # 锐化，增强图像的高频分量。
    m = nn.ReflectionPad2d(1)
    x = k.filters.gaussian_blur2d(input, (3, 3), (1.5, 1.5))
    x = input + (input - x)
    return x
def Ubo(input):
    x = k.filters.box_blur(input, (3, 3))
    x = input + (input - x)
    return x
def Uma(input):
    m = nn.ReflectionPad2d(1)
    x = k.filters.max_blur_pool2d(input,kernel_size=2,)
    x = m(x)
    x = input + (input - x)
    return x
def Ume(input):
    x = k.filters.median_blur(input,(3,3))
    x = input + (input - x)
    return x
def UL(input):
    x = k.filters.laplacian(input, kernel_size=3)
    x = input + x
    return x
def US(input):
    x = k.filters.sobel(input)
    x = input + x
    return x
def UC(input):
    x = k.filters.canny(input)
    x = input + x
    return x
def ULoG(input):
    x = LoG(input)
    x = input + x
    return x


def IG(input):
    m = nn.ReflectionPad2d(1)
    x = k.filters.gaussian_blur2d(input, (3, 3), (1.5, 1.5))
    x = input - x
    return x
def Ibo(input):
    x = k.filters.box_blur(input, (3, 3))
    x = input - x
    return x
def Ima(input):
    m = nn.ReflectionPad2d(1)
    x = k.filters.max_blur_pool2d(input,kernel_size=3)
    x = m(x)
    x = input - x
    return x
def Ime(input):
    x = k.filters.median_blur(input,(3,3))
    x = input - x
    return x
def IL(input):
    x = k.filters.laplacian(input, kernel_size=3)
    # x = input + x
    return x
def IS(input):
    x = k.filters.sobel(input)
    # x = input + x
    return x
def IC(input):
    x = k.filters.canny(input)
    # x = input + x
    return x
def ILoG(input):
    x = LoG(input)
    # x = input + x
    return x


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel).to(device)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel).to(device)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel).to(device) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel).to(device) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel).to(device) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).to(
        device)
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1).to(device)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).to(device)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(device)
    return window


class TV_Loss(nn.Module):
    """docstring for TV_Loss"""

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, vi_images, ir_images, output_images):

        # input_images = [input_images[i] for i in input_images]
        # fusion_images = output_images['Fusion']
        fusion_images = output_images
        tv_loss = 0
        input_image1 = vi_images
        input_image2 = ir_images
        H1, W1 = input_image1.shape[2], input_image1.shape[3]
        R1 = input_image1 - fusion_images
        L_vi = torch.pow(R1[:, :, 1:H1, :] - R1[:, :, 0:H1 - 1, :], 2).sum() + \
               torch.pow(R1[:, :, :, 1:W1] - R1[:, :, :, 0:W1 - 1], 2).sum()
        H2, W2 = input_image2.shape[2], input_image2.shape[3]
        R2 = input_image2 - fusion_images
        L_ir = torch.pow(R2[:, :, 1:H2, :] - R2[:, :, 0:H2 - 1, :], 2).sum() + \
               torch.pow(R2[:, :, :, 1:W2] - R2[:, :, :, 0:W2 - 1], 2).sum()
        tv_loss = 0.01*L_vi+0.01*L_ir

        return tv_loss


class VIF_SSIM_Loss(nn.Module):
    """docstring for VIF_SSIM_Loss"""

    def __init__(self, kernal_size=11, num_channels=1, C=9e-4, device='cuda:0'):
        super(VIF_SSIM_Loss, self).__init__()
        self.kernal_size = kernal_size
        self.num_channels = num_channels
        self.device = device
        self.c = C

        self.avg_kernal = torch.ones(num_channels, 1, self.kernal_size, self.kernal_size) / (self.kernal_size) ** 2
        self.avg_kernal = self.avg_kernal.to(device)

    def forward(self, vi_images,ir_images, output_images):
        # vis_images, inf_images, fusion_images = input_images[self.sensors[0]], input_images[self.sensors[1]], \
        #                                         output_images['Fusion']
        vis_images, inf_images, fusion_images = vi_images, ir_images,  output_images
        batch_size, num_channels = vis_images.shape[0], vis_images.shape[1]

        vis_images_mean = F.conv2d(vis_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
        vis_images_var = torch.abs(F.conv2d(vis_images ** 2, self.avg_kernal, stride=self.kernal_size,
                                            groups=num_channels) - vis_images_mean ** 2)

        inf_images_mean = F.conv2d(inf_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
        inf_images_var = torch.abs(F.conv2d(inf_images ** 2, self.avg_kernal, stride=self.kernal_size,
                                            groups=num_channels) - inf_images_mean ** 2)

        fusion_images_mean = F.conv2d(fusion_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
        fusion_images_var = torch.abs(F.conv2d(fusion_images ** 2, self.avg_kernal, stride=self.kernal_size,
                                               groups=num_channels) - fusion_images_mean ** 2)

        vis_fusion_images_var = F.conv2d(vis_images * fusion_images, self.avg_kernal, stride=self.kernal_size,
                                         groups=num_channels) - vis_images_mean * fusion_images_mean
        inf_fusion_images_var = F.conv2d(inf_images * fusion_images, self.avg_kernal, stride=self.kernal_size,
                                         groups=num_channels) - inf_images_mean * fusion_images_mean

        C = torch.ones_like(fusion_images_mean) * self.c

        ssim_l_vis_fusion = (2 * vis_images_mean * fusion_images_mean + C) / \
                            (vis_images_mean ** 2 + fusion_images_mean ** 2 + C)
        ssim_l_inf_fusion = (2 * inf_images_mean * fusion_images_mean + C) / \
                            (inf_images_mean ** 2 + fusion_images_mean ** 2 + C)

        ssim_s_vis_fusion = (vis_fusion_images_var + C) / (vis_images_var + fusion_images_var + C)
        ssim_s_inf_fusion = (inf_fusion_images_var + C) / (inf_images_var + fusion_images_var + C)

        score_vis_inf_fusion = (vis_images_mean > inf_images_mean) * ssim_l_vis_fusion * ssim_s_vis_fusion + \
                               (vis_images_mean <= inf_images_mean) * ssim_l_inf_fusion * ssim_s_inf_fusion

        ssim_loss = 1 - score_vis_inf_fusion.mean()

        return ssim_loss


def imread(path):
    # img = cv2.imread(path,cv2.IMREAD_COLOR)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img[:, :, 0]
    # return img




def prepare_data(config, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
      :param dataset:
      :param config:
    """
    if config.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        # data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        # 将图片按序号排序
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    # print(data)

    return data


def make_data(config, data, label, data_dir):
    """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
    if config.is_train:
        # savepath = os.path.join(os.getcwd(), os.path.join('checkpoint',data_dir,'train.h5'))
        savepath = os.path.join('.', os.path.join(config.checkpoint_dir, data_dir, 'train.h5'))
        if not os.path.exists(os.path.join('.', os.path.join(config.checkpoint_dir, data_dir))):
            os.makedirs(os.path.join('.', os.path.join(config.checkpoint_dir, data_dir)))
    else:
        savepath = os.path.join('.', os.path.join(config.checkpoint_dir, data_dir, 'test.h5'))
        if not os.path.exists(os.path.join('.', os.path.join(config.checkpoint_dir, data_dir))):
            os.makedirs(os.path.join('.', os.path.join(config.checkpoint_dir, data_dir)))
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
        data = prepare_data(config, dataset=data_dir)
    else:
        data = prepare_data(config, dataset=data_dir)

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6

    if config.is_train:
        for i in range(len(data)):

            # input_, label_ = preprocess(data[i], config.scale)
            input_ = (imread(data[i]) - 127.5) / 127.5
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
            nx += 1;
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
    make_data(config, arrdata, arrlabel, data_dir)

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
        label = np.array(hf.get('label'))
        return data, label
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * \
                                  self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * \
                                  self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


