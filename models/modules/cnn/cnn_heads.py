# -*-coding:utf8-*-
import torch
from .tensor_op import pixel_shuffle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points
        快速非最大抑制去除附近的点
    """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    '''当该点得分在以自身为中心的9*9范围内最大则为True，得到初代特征点'''
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        '''把max_mask中为True的点的周围9*9范围的值都变成True'''
        supp_scores = torch.where(supp_mask, zeros, scores)
        '''把supp_mask中为True的点置零，False的点取原scores对应的值'''
        new_max_mask = supp_scores == max_pool(supp_scores)
        '''和第一步作用类似，这里整体的作用是将 初代特征点 范围内的得分都置零以后，用剩余范围里的得分得到二代特征点'''
        max_mask = max_mask | (new_max_mask & (~supp_mask))
        '''合并初代特征点与二代特征点'''
    return torch.where(max_mask, scores, zeros)

def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border
        删除太靠近边界的关键点
    """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


class DetectorHead(torch.nn.Module):
    def __init__(self, config, input_channel, grid_size, using_bn=True):
        super(DetectorHead, self).__init__()
        self.config = config
        self.grid_size = grid_size
        self.using_bn = using_bn
        ##
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(256, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)

        self.bnPa, self.bnPb = None, None
        if using_bn:
            self.bnPa = torch.nn.BatchNorm2d(256)
            self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, curr_max_kp, curr_key_thresh):

        out = None
        if self.using_bn:
            out = self.bnPa(self.relu(self.convPa(x)))
            out = self.bnPb(self.convPb(out))   # (B,65,H,W)
        else:
            out = self.relu(self.convPa(x))
            out = self.convPb(out)              # (B,65,H,W)

        scores = torch.nn.functional.softmax(out, 1)[:, :-1]

        # i = 0
        # while i < 3:
        #     x_4 = scores.detach().cpu().numpy()
        #     x_4 = torch.tensor(x_4, dtype=torch.float)
        #     img = plt.imshow(x_4.detach()[0][i])
        #     img.axes.xaxis.set_visible(False)
        #     img.axes.yaxis.set_visible(False)
        #     plt.show()
        #     matplotlib.use('TkAgg')
        #     i = i + 1

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, 4)

        # Extract keypoints; 提取关键点;
        keypoints = [
            torch.nonzero(s > curr_key_thresh)
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders; 丢弃图像边界附近的关键点;
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score; 保留得分最高的k个关键点;
        if curr_max_kp >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, curr_max_kp)
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y); 转换;
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        return keypoints, scores, h, w


class DescriptorHead(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grid_size, using_bn=True):
        super(DescriptorHead, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn

        self.convDa = torch.nn.Conv2d(input_channel, 256, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convDb = torch.nn.Conv2d(256, output_channel, kernel_size=1, stride=1, padding=0)

        self.bnDa, self.bnDb = None, None
        if using_bn:
            self.bnDa = torch.nn.BatchNorm2d(256)
            self.bnDb = torch.nn.BatchNorm2d(output_channel)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnDa(self.relu(self.convDa(x)))
            out = self.bnDb(self.convDb(out))
        else:
            out = self.relu(self.convDa(x))
            out = self.convDb(out)

        # out_norm = torch.norm(out, p=2, dim=1)  # Compute the norm.
        # out = out.div(torch.unsqueeze(out_norm, 1))  # Divide by norm to normalize.

        # TODO: here is different with tf.image.resize_bilinear
        desc = F.interpolate(out, scale_factor=self.grid_size, mode='bilinear', align_corners=False)
        # i = 0
        # while i < 5:
        #     x_4 = desc.detach().cpu().numpy()
        #     x_4 = torch.tensor(x_4, dtype=torch.float)
        #     img = plt.imshow(x_4.detach()[0][i])
        #     img.axes.xaxis.set_visible(False)
        #     img.axes.yaxis.set_visible(False)
        #     plt.show()
        #     matplotlib.use('TkAgg')
        #     i = i + 1

        desc = F.normalize(desc, p=2, dim=1)                                                            # normalize by channel
        # i = 0
        # while i < 5:
        #     x_4 = desc.detach().cpu().numpy()
        #     x_4 = torch.tensor(x_4, dtype=torch.float)
        #     img = plt.imshow(x_4.detach()[0][i])
        #     img.axes.xaxis.set_visible(False)
        #     img.axes.yaxis.set_visible(False)
        #     plt.show()
        #     matplotlib.use('TkAgg')
        #     i = i + 1

        # return {'desc_raw': out, 'desc': desc}

        return desc
