# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 16:01
# @Author  : Mina Han
# @FileName: model_att.py
# @Software: PyCharm
# !/usr/bin/env python
# -*-coding:utf-8 -*-
# import torch

from utils import *
import os
from Generator_ss import G
from torch.utils.tensorboard import SummaryWriter
from feature import feature
from Dis_res import D
import numpy.ma as ma
from attention_trans import *

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CUDA_VISIBLE_DEVICES = 0
class FusionGAN:
    """融合生成对抗网络"""
    def __init__(self, config):
        self.config = config
        self.gen = G().to(device)
        self.dis = D().to(device)
        self.fea = feature().to(device)
        # self.att=EMAU(1, 1).cuda()
        # self.trans=trans().to(device)

        self.gen_op = torch.optim.Adam(self.gen.parameters(), lr=config.learning_rate)
        self.dis_op = torch.optim.Adam(self.dis.parameters(), lr=config.learning_rate)
        self.fea_op = torch.optim.Adam(self.fea.parameters(), lr=config.learning_rate)
        # self.att_op=torch.optim.Adam(self.att.parameters(),lr=config.learning_rate)
        # self.trans_op=torch.optim.Adam(self.trans.parameters(),lr=config.learning_rate)


    def dis_loss(self, vi_pro, ir_pro, fusion_pro):
        ############dis and dis2###########
        # print(vi_pro.is_cuda,fusion_pro_1.is_cuda)   #判断两个数据是否再GPU
        self.d_loss_1 = torch.mean(torch.square(vi_pro - torch.Tensor(vi_pro.shape).uniform_(0.7, 1.2).to(device)))
        # print(self.d_loss_1)
        self.d_loss_2 = torch.mean(torch.square(ir_pro - torch.Tensor(ir_pro.shape).uniform_(0.6, 1.1).to(device)))
        self.d_loss_3 = torch.mean(torch.square(fusion_pro - torch.Tensor(fusion_pro.shape).uniform_(0., 0.3).to(device)))
        self.d_loss = self.d_loss_1 + self.d_loss_2 + self.d_loss_3
        return self.d_loss

    # def gen_loss(self,fusion_out,fusion_pro,ir_labels,vi_labels,ir_labels_mask,vi_labels_mask):
    def gen_loss(self, fusion_out, fusion_pro, ir_labels, vi_labels, ir_labels_mask):

        #############Against loss############
        self.g_loss_1=torch.mean(torch.square(0.5 * fusion_pro-torch.Tensor(fusion_pro.shape).uniform_(0.7, 1.2).to(device)) \
                                                + 0.5 * torch.square(fusion_pro-torch.Tensor(fusion_pro.shape).uniform_(0.6, 1.1).to(device)))

        #############Content loss###########
        # self.g_loss_2=torch.mean(torch.square(fusion_out-ir_labels))+5*torch.mean(torch.square(gradient(fusion_out)-gradient(vi_labels)))

        ##########add entropy loss###########
        self.g_loss_ent=torch.mean(torch.nn.functional.softmax(fusion_out) * torch.log(torch.nn.functional.softmax(torch.abs(vi_labels))))

        ###########add Traditional loss##########
        self.g_loss_tradi = torch.mean(torch.square(fusion_out - 0.5 * (vi_labels + ir_labels)))

        ###########new content###############
        self.g_loss_2 = torch.mean(torch.square(fusion_out - ir_labels) + torch.square(fusion_out - vi_labels)) + \
                                        5 * torch.mean(torch.square(gradient(fusion_out) - gradient(vi_labels)) + torch.square(gradient(fusion_out)-gradient(ir_labels)))

        # self.g_loss=self.g_loss_1+100*self.g_loss_2 + 50*self.g_loss_ent
        #########1.18 new all loss###########
        # 1.3 模型中的全局信息损失函数的各部分比例
        # self.g_loss = self.g_loss_1 + 50 * self.g_loss_2 + self.g_loss_ent + 50*self.g_loss_tradi

        # 1.4 模型中的全局信息损失函数的各部分比例
        self.g_loss = self.g_loss_1 + 50 * self.g_loss_2 + 30 * self.g_loss_ent + 50 * self.g_loss_tradi

        #########add Semantic segmentation loss##########
        #这里计算了ir和vi的欧式距， 后期可以改成对于vi来讲，计算梯度信息
        self.g_loss_ss_ir = torch.mean(torch.square((fusion_out * ir_labels_mask) - (ir_labels * ir_labels_mask)))
        # self.g_loss_ss_vi = torch.mean(torch.square((fusion_out * vi_labels_mask) - (vi_labels * vi_labels_mask)))
        # self.g_loss_ss_vi = torch.mean(torch.square(gradient(fusion_out * vi_labels_mask) - gradient(vi_labels * vi_labels_mask)))
        self.g_loss_ss = 100 * self.g_loss_ss_ir
        print('g_losss_ss = ', self.g_loss_ss)


        #### 4.11 加入语义分割损失的生成器的总的损失#####
        self.g_loss = 0.5 * self.g_loss + 0.5 * self.g_loss_ss

        return self.g_loss


    def train_step(self, ir_imgs, ir_labels, vi_imgs, vi_labels, ir_mask, ir_labels_mask, k=2):
        self.gen.train()

        d_loss_val = 0
        g_loss_val = 0
        fusion_img = self.gen(ir_imgs, vi_imgs)

        with torch.no_grad():
            fusion_out = fusion_img
        for _ in range(k):
            self.dis_op.zero_grad()
            vi_pro = self.dis(vi_labels)
            ir_pro = self.dis(ir_labels)
            fusion_pro = self.dis(fusion_out)
            dis_loss = self.dis_loss(vi_pro, ir_pro, fusion_pro)

            d_loss_val = d_loss_val + dis_loss.cpu().item()
            dis_loss.backward(retain_graph=True)
            self.dis_op.step()

        self.gen_op.zero_grad()

        fusion_pro = self.dis(fusion_out)
        # g_loss=self.gen_loss(fusion_out,fusion_pro,ir_labels,vi_labels,ir_labels_mask,vi_labels_mask)
        g_loss = self.gen_loss(fusion_out, fusion_pro, ir_labels, vi_labels, ir_labels_mask)
        g_loss_val = g_loss_val+g_loss.cpu().item()
        g_loss.backward(retain_graph=False)
        self.gen_op.step()

        return d_loss_val/k, g_loss_val

    def train(self):
        if self.config.is_train:
            input_setup(self.config, "I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_ir")
            input_setup(self.config, "I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_ir_mask")
            input_setup(self.config, "I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_vi")
            # input_setup(self.config, "Train_vi_mask")

            #####训练彩色图像#####
            # input_setup(self.config, "E:/data/multi-focus_image_datasets/a_1_image")
            # input_setup(self.config, "E:/data/multi-focus_image_datasets/b_1_image")
        else:
            nx_ir, ny_ir = input_setup(self.config, "I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_ir")
            nx_vi, ny_vi = input_setup(self.config, "I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_vi")

        if self.config.is_train:
            data_dir_ir = os.path.join("I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_ir\\train.h5")
            data_dir_vi = os.path.join("I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_vi\\train.h5")
            data_dir_ir_mask = os.path.join("I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_ir_mask\\train.h5")
            # data_dir_vi_mask = os.path.join('./{}'.format(self.config.checkpoint_dir), "Train_vi_mask", "train.h5")
        else:
            data_dir_ir = os.path.join("I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_ir\\test.h5")
            data_dir_vi = os.path.join("I:\\PRAI\\CODE\\CODE\\fusion\\TLGAN\\data\\Train_vi\\test.h5")

        # if self.config.is_train:
        #     data_dir_ir = os.path.join('./{}'.format(self.config.checkpoint_dir), "Train_ir", "train.h5")
        #     data_dir_vi = os.path.join('./{}'.format(self.config.checkpoint_dir), "Train_vi", "train.h5")
        #     data_dir_ir_mask = os.path.join('./{}'.format(self.config.checkpoint_dir), "Train_ir_mask", "train.h5")
        #     # data_dir_vi_mask = os.path.join('./{}'.format(self.config.checkpoint_dir), "Train_vi_mask", "train.h5")
        # else:
        #     data_dir_ir = os.path.join('./{}'.format(self.config.checkpoint_dir), "Test_ir", "test.h5")
        #     data_dir_vi = os.path.join('./{}'.format(self.config.checkpoint_dir), "Test_vi", "test.h5")

        train_data_ir, train_label_ir = read_data(data_dir_ir)
        train_data_vi, train_label_vi = read_data(data_dir_vi)
        train_data_ir_mask, train_label_ir_mask = read_data(data_dir_ir_mask)                                           # 消耗内存
        # train_data_vi_mask, train_label_vi_mask = read_data(data_dir_vi_mask)

        random_index = torch.randperm(len(train_data_ir))
        train_data_vi = train_data_vi[random_index]
        train_data_ir = train_data_ir[random_index]
        train_label_vi = train_label_vi[random_index]
        train_label_ir = train_label_ir[random_index]
        train_data_ir_mask = train_data_ir_mask[random_index]
        train_label_ir_mask = train_label_ir_mask[random_index]
        # train_data_vi_mask = train_data_vi_mask[random_index]
        # train_label_vi_mask = train_label_vi_mask[random_index]

        batch_size = self.config.batch_size

        if self.config.is_train:
            with SummaryWriter(self.config.summary_dir) as writer:
                epochs = self.config.epoch
                for epoch in range(1, 1+epochs):
                    batch_idxs = len(train_data_ir) // self.config.batch_size
                    d_loss_mean = 0
                    g_loss_mean = 0
                    for idx in range(1, 1+batch_idxs):
                        start_idx = (idx - 1) * batch_size
                        ir_images = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        ir_labels = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vi_images = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vi_labels = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        ir_mask = train_data_ir_mask[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        ir_labels_mask = train_label_ir_mask[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        # vi_mask = train_data_vi_mask[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        # vi_labels_mask = train_label_vi_mask[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

                        ir_images = torch.tensor(ir_images).float().to(device)
                        ir_labels = torch.tensor(ir_labels).float().to(device)
                        vi_images = torch.tensor(vi_images).float().to(device)
                        vi_labels = torch.tensor(vi_labels).float().to(device)
                        ir_mask = torch.tensor(ir_mask).float().to(device)
                        ir_labels_mask = torch.tensor(ir_labels_mask).float().to(device)
                        # vi_mask = torch.tensor(vi_mask).float().to(device)
                        # vi_labels_mask = torch.tensor(vi_labels_mask).float().to(device)

                        d_loss, g_loss = self.train_step(ir_images, ir_labels, vi_images,vi_labels,ir_mask,ir_labels_mask, 2)
                        d_loss_mean += d_loss
                        g_loss_mean += g_loss

                        print('Epoch {}/{}, Step {}/{}, gen_loss = {:.4f},  dis_loss = {:.4f}'.format(epoch, epochs, idx, batch_idxs, g_loss, d_loss))
                    model_path = os.path.join(os.getcwd(), 'checkpoint', 'epoch{}'.format(epoch))
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    if epoch < 10:
                        torch.save(self.gen.state_dict(), 'checkpoint/epoch{}'.format(epoch)+'/'+'model-0{}'.format(epoch)+'.pth')
                    elif epoch < 100:
                        torch.save(self.gen.state_dict(),
                                   'checkpoint/epoch{}'.format(epoch) + '/' + '{}'.format(epoch) + '.pth')

                    d_loss_mean = d_loss_mean / batch_idxs
                    g_loss_mean = g_loss_mean / batch_idxs
                    writer.add_scalar('scalar/gen_loss', g_loss_mean, epoch)
                    writer.add_scalar('scalar/dis_loss', d_loss_mean, epoch)

            print('Saving model......')
            torch.save(self.gen.state_dict(), '%s/model-final.pth' % (self.config.checkpoint_dir))
            print("Training Finished, Total EPOCH = %d" % self.config.epoch)