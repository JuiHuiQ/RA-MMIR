# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : train.py

Author     ：yujing_rao
"""
import logging
import os
import random

import torch
from kornia.losses import SSIMLoss
from torch import nn

from Loss.D_loss import DLoss
from Loss.measureG import CLoss, GLoss
from python.iqa import qem
from modules.discriminator import Discriminator
from modules.generator import Generator
from utils.read_h5 import read_data
from utils.wirte_h5 import  input_setup


class Train:
    # defined a train process, run
    def __init__(self,environment_probe, config):
        logging.info(f'ATFusionGAN training process!')
        self.config=config
        self.environment_probe=environment_probe

        logging.info(f'generator module define!')
        self.generator=Generator()
        logging.info(f'discriminator module define!')
        self.discriminator=Discriminator()

        logging.info(f'learning rate define!')
        self.opt_generator=torch.optim.Adam(self.generator.parameters(),lr=config.learning_rate)
        self.opt_discriminator=torch.optim.Adam(self.discriminator.parameters(),lr=config.learning_rate)

        logging.info(f'assign device:{environment_probe.device} to modules!')
        self.generator.to(environment_probe.device)
        self.discriminator.to(environment_probe.device)

        logging.info(f'loss function define!')
        # generator M_content
        self.C_loss_ob=CLoss()
        self.G_loss_ob=GLoss(device=self.environment_probe.device)
        # generator M_structure
        self.SSIM_loss_ob=SSIMLoss(window_size=11, reduction='none')
        self.d_loss_ob=DLoss()

        #loss to cuda
        self.C_loss_ob.cuda()
        self.G_loss_ob.cuda()
        self.SSIM_loss_ob.cuda()
        self.d_loss_ob.cuda()
        self.quality_aware=0.6


    def train_step(self,ir_images,ir_labels,vi_images,vi_labels):
        logging.debug('train generator')
        self.generator.train()
        d_loss=0
        g_loss=0
        fused_img=self.generator(ir_images,vi_images)

        with torch.no_grad():
            fusion_out=fused_img
        logging.debug('train discriminator')
        # weights = torch.load('checkpoint/epoch56/discriminator_model-056.pt')
        # self.discriminator.load_state_dict(weights)
        # self.discriminator.eval()
        for d_train_number in range(2):
            real_ir=ir_labels
            real_vi=vi_labels
            self.quality_aware=qem(ir_labels,vi_labels)
            self.opt_discriminator.zero_grad()
            real_ir_pro=self.discriminator(real_ir)
            real_vi_pro=self.discriminator(real_vi)
            fake_fused_pro=self.discriminator(fusion_out)
            d1_loss=self.d_loss_ob(real_vi_pro,1,0)+self.d_loss_ob(real_ir_pro,0,1)
            d2_loss=self.d_loss_ob(fake_fused_pro,0,0)
            d_train_loss=d1_loss+d2_loss
            d_loss+=d_train_loss.cpu().item()
            d_train_loss.backward(retain_graph=True)
            self.opt_discriminator.step()
        self.opt_generator.zero_grad()
        #loss
        # quality-aware parameter
        self.quality_aware = qem(ir_labels,vi_labels)
        loss_measure=self.quality_aware*(5*self.G_loss_ob(fusion_out,vi_images)+10*self.C_loss_ob(fusion_out,vi_images))+(1-self.quality_aware)*(5*self.G_loss_ob(fusion_out,ir_images)+10*self.C_loss_ob(fusion_out,ir_images))
        loss_structure=(self.quality_aware*self.SSIM_loss_ob(fusion_out,ir_images)+(1-self.quality_aware)*self.SSIM_loss_ob(fusion_out,vi_images)).mean()
        #adverse loss
        self.discriminator.eval()
        fused_img_pro=self.discriminator(fusion_out)
        g_loss_adv=self.d_loss_ob(fused_img_pro,random.uniform(0.7,1) ,random.uniform(0.7,1) )
        # gc_train_loss=loss_measure+0.1*loss_structure
        gc_train_loss = loss_measure+loss_structure
        g_loss=gc_train_loss+0.4*g_loss_adv
        #generator backward
        g_loss.backward()
        self.opt_generator.step()

        # logging.debug('train generator')
        # self.discriminator.train()
        # self.generator.eval()
        return g_loss,d_loss/2


    def run(self):
        # train dataset
        logging.info(f'dataset define!')

        # # img--->h5 file
        # input_setup(self.config, "Train_ir",ir_flag=True)
        # input_setup(self.config, "Train_vi",ir_flag=False)

        # address of h5 file
        data_dir_ir = os.path.join('{}'.format(self.config.checkpoint_dir), "Train_ir", "train.h5")
        data_dir_vi = os.path.join('{}'.format(self.config.checkpoint_dir), "Train_vi", "train.h5")

        # read h5 file
        train_data_ir, train_label_ir = read_data(data_dir_ir)
        train_data_vi, train_label_vi = read_data(data_dir_vi)

        # random_set
        # random_index = torch.randperm(len(train_data_ir))
        #
        # train_data_vi = train_data_vi[random_index]
        # train_data_ir = train_data_ir[random_index]
        # train_label_vi = train_label_vi[random_index]
        # train_label_ir = train_label_ir[random_index]

        batch_size=self.config.batch_size
        # g_loss_best=10
        # d_loss_best=10
        # g1_loss_best = 10
        # d1_loss_best = 10
        for epoch in range(1,self.config.epoch+1):
            batch_idxs=len(train_data_ir)//self.config.batch_size
            # d_loss_sum=0
            # g_loss_sum=0
            for idx in range(1,1+batch_idxs):
                start_idx = (idx - 1) * batch_size
                ir_images = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                ir_labels = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                vi_images = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                vi_labels = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

                ir_images = torch.tensor(ir_images).float().to(self.environment_probe.device)
                ir_labels = torch.tensor(ir_labels).float().to(self.environment_probe.device)
                vi_images = torch.tensor(vi_images).float().to(self.environment_probe.device)
                vi_labels = torch.tensor(vi_labels).float().to(self.environment_probe.device)
                # return g_loss(all)，g1_train_loss
                g_loss,d_loss=self.train_step(ir_images,ir_labels,vi_images,vi_labels)

                print('Epoch {}/{}, Step {}/{}, gen_loss = {:.4f}, gen_loss = {:.4f}'.format(epoch, self.config.epoch, idx,
                                                                                              batch_idxs, g_loss,d_loss))
            model_path = os.path.join(os.getcwd(), '../checkpoint', 'epoch_{}'.format(epoch))

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.generator.state_dict(),
                       '../checkpoint/epoch_{}'.format(epoch) + '/' + 'model-{}'.format(epoch) + '.pt')

        print('Saving final model......')
        torch.save(self.generator.state_dict(), '%s/model-final.pt' % (self.config.checkpoint_dir))
        # torch.save(self.discriminator.state_dict(),'%s/model_discriminator-final.pt' % (self.config.checkpoint_dir))
        print("Training Finished, Total EPOCH = %d" % self.config.epoch)









