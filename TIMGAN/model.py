# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : model.py
Create on  ：2021/7/26 14:03

Author     ：yujing_rao
"""
import torch
from utils import *
import os
from Gfusion import Gen
from Discrimitor import Dv,Di
from torch.utils.tensorboard import SummaryWriter
from adabelief_pytorch import AdaBelief
import kornia as K

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# device = 'cpu'
# device ='cuda'
class DenseGAN:
    def __init__(self,config):
        self.config=config
        self.gen=Gen().to(device)
        self.dis1=Dv().to(device)
        self.dis2=Di().to(device)

        self.TVloss = TV_Loss().to(device)
        self.VIloss = VIF_SSIM_Loss().to(device)


        self.gen_op=AdaBelief(self.gen.parameters(), lr=config.learning_rate, rectify=False, print_change_log=False)
        self.dis_op1=AdaBelief(self.dis1.parameters(), lr=config.learning_rate, rectify=False, print_change_log=False)
        self.dis_op2=AdaBelief(self.dis2.parameters(), lr=config.learning_rate, rectify=False, print_change_log=False)
        self.percep = PerceptualLoss(layer_weights={'conv5_4': 1.}).to(device)



    def dis_loss(self,vi_pro,fusion_pro1, fusion_pro2, ir_pro):

        eps = 1e-8
        Ra_loss_rf1 = torch.sigmoid((vi_pro) - torch.mean(fusion_pro1, dim=0))
        Ra_loss_rf2 = torch.sigmoid((ir_pro) - torch.mean(fusion_pro2, dim=0))
        Ra_loss_fr1 = torch.sigmoid((fusion_pro1) - torch.mean(vi_pro, dim=0))
        Ra_loss_fr2 = torch.sigmoid((fusion_pro2) - torch.mean(ir_pro, dim=0))
        self.L_Ra_d2 = - torch.mean(torch.log(Ra_loss_rf2+eps)) - torch.mean(torch.log(1 - Ra_loss_fr2+eps))
        self.L_Ra_d1 = - torch.mean(torch.log(Ra_loss_rf1+eps)) - torch.mean(torch.log(1 - Ra_loss_fr1+eps))
        return self.L_Ra_d2 + self.L_Ra_d1


    def gen_loss(self,fusion_out,fusion_pro1,fusion_pro2 , ir_labels,vi_labels,vi_pro,ir_pro,img_sum):


        eps = 1e-8
        '''adversarial loss'''
        Ra_loss_rf1 = torch.sigmoid((vi_pro) - torch.mean(fusion_pro1, dim=0))
        Ra_loss_rf2 = torch.sigmoid((ir_pro) - torch.mean(fusion_pro2, dim=0))
        Ra_loss_fr1 = torch.sigmoid((fusion_pro1) - torch.mean(vi_pro, dim=0))
        Ra_loss_fr2 = torch.sigmoid((fusion_pro2) - torch.mean(ir_pro, dim=0))
        self.L_Ra_g1 = - torch.mean(torch.log(1 - Ra_loss_rf1+eps)) - torch.mean(torch.log(Ra_loss_fr1+eps))
        self.L_Ra_g2 = - torch.mean(torch.log(1 - Ra_loss_rf2+eps)) - torch.mean(torch.log(Ra_loss_fr2+eps))


        self.g_loss_2=1*torch.mean(torch.square((Laplacion(fusion_out))-(Laplacion(ir_labels))))+9*torch.mean(torch.square(Laplacion((fusion_out))-Laplacion((vi_labels))))

        self.VIssim = self.VIloss(vi_labels,ir_labels,fusion_out)
        self.TV = self.TVloss(vi_labels,ir_labels,fusion_out)

        self.g_loss=1*self.L_Ra_g1+1*self.L_Ra_g2+5 * self.g_loss_2+self.VIssim+self.TV


        return self.g_loss

    def train_step(self, ir_imgs, ir_labels, vi_imgs, vi_labels,k=2):
        self.gen.train()

        d_loss_val=0
        g_loss_val=0

        fusion_img=self.gen(ir_imgs,vi_imgs)
        ir = IR_test(ir_labels)
        vi = Unsharp_mask(vi_labels)
        img_sum = ir+vi



        with torch.no_grad():

            fusion_out=fusion_img

        for _ in range(k):
            self.dis_op1.zero_grad()
            self.dis_op2.zero_grad()

            vi_pro=self.dis1(vi_labels)
            ir_pro=self.dis2(img_sum)

            fusion_pro1=self.dis1(fusion_out)
            fusion_pro2=self.dis2(fusion_out)
            dis_loss=self.dis_loss(vi_pro,fusion_pro1,fusion_pro2, ir_pro)
            # dis_loss=self.dis_loss(vi_pro,fusion_pro1)

            d_loss_val =d_loss_val+ dis_loss.to(device).item()
            dis_loss.backward(retain_graph=True)
            self.dis_op1.step()
            self.dis_op2.step()


        self.gen_op.zero_grad()

        g_loss=self.gen_loss(fusion_out,fusion_pro1, fusion_pro2, ir_labels,vi_labels,vi_pro,ir_pro,img_sum)

        # ,vi_pro)
        g_loss_val=g_loss_val+g_loss.cuda().item()
        g_loss.backward(retain_graph=False)
        self.gen_op.step()

        return d_loss_val/k,g_loss_val

    def train(self):
        if self.config.is_train:
            input_setup( self.config, "Train_ir")
            input_setup(self.config, "Train_vi")
        else:
            nx_ir, ny_ir = input_setup(self.config, "Test_ir")
            nx_vi, ny_vi = input_setup(self.config, "Test_vi")

        if self.config.is_train:
            data_dir_ir = os.path.join('./{}'.format(self.config.checkpoint_dir), "Train_ir", "train.h5")
            data_dir_vi = os.path.join('./{}'.format(self.config.checkpoint_dir), "Train_vi", "train.h5")
        else:
            data_dir_ir = os.path.join('./{}'.format(self.config.checkpoint_dir), "Test_ir", "test.h5")
            data_dir_vi = os.path.join('./{}'.format(self.config.checkpoint_dir), "Test_vi", "test.h5")

        train_data_ir, train_label_ir = read_data(data_dir_ir)
        train_data_vi, train_label_vi = read_data(data_dir_vi)

        random_index = torch.randperm(len(train_data_ir))
        train_data_vi = train_data_vi[random_index]
        train_data_ir = train_data_ir[random_index]
        train_label_vi = train_label_vi[random_index]
        train_label_ir = train_label_ir[random_index]
        batch_size = self.config.batch_size

        if self.config.is_train:
            with SummaryWriter(self.config.summary_dir) as writer:


                epochs = self.config.epoch
                for epoch in range(1,1+epochs):
                    print(len(train_data_ir))
                    batch_idxs = len(train_data_ir) // self.config.batch_size
                    d_loss_mean = 0
                    g_loss_mean = 0
                    for idx in range(1,1+batch_idxs):
                        start_idx = (idx - 1) * batch_size
                        ir_images = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        ir_labels = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vi_images = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vi_labels = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

                        ir_images= torch.tensor(ir_images).float().to(device)
                        ir_labels = torch.tensor(ir_labels).float().to(device)
                        vi_images  = torch.tensor(vi_images).float().to(device)
                        vi_labels = torch.tensor(vi_labels).float().to(device)

                        d_loss, g_loss = self.train_step(ir_images, ir_labels, vi_images,vi_labels, 2)
                        d_loss_mean += d_loss
                        g_loss_mean += g_loss
                        print('Epoch {}/{}, Step {}/{}, gen_loss = {:.4f},  dis_loss = {:.4f}'.format(epoch, epochs,idx, batch_idxs,g_loss, d_loss))
                    model_path=os.path.join(os.getcwd(), self.config.checkpoint_dir, 'epoch{}'.format(epoch))
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    if epoch<=10:
                        torch.save(self.gen.state_dict(), self.config.checkpoint_dir+'/epoch{}'.format(epoch)+'/'+'model-0{}'.format(epoch)+'.pth')
                    elif epoch<100:
                        torch.save(self.gen.state_dict(), self.config.checkpoint_dir+'/epoch{}'.format(epoch) + '/' + '{}'.format(epoch) + '.pth')

                    d_loss_mean =d_loss_mean / batch_idxs
                    g_loss_mean =g_loss_mean / batch_idxs
                    writer.add_scalar('scalar/gen_loss', g_loss_mean, epoch)
                    writer.add_scalar('scalar/dis_loss', d_loss_mean, epoch)

            print('Saving model......')
            torch.save(self.gen.state_dict(), '%s/model-final.pth' % (self.config.checkpoint_dir))
            print("Training Finished, Total EPOCH = %d" % self.config.epoch)