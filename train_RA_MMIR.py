import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.distributed as dist
import yaml
import time
from pathlib import Path
import torch.optim as optim
import sys
import numpy as np

# from skimage.metrics import structural_similarity as ssim

from models.RAMM_Point_bn import RAMM_Point
from models.RA_MMIR import RA_MMIR

from utils.common import increment_path, init_seeds, clean_checkpoint, reduce_tensor, download_base_files, debug_image_plot, time_synchronized, test_model, ModelEMA
from utils.preprocess_utils import torch_find_matches
from utils.dataset import COCO_loader, COCO_valloader, collate_batch
from torch.utils.tensorboard import SummaryWriter

import lpips

# torch.cuda.set_device(1)

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def change_lr(epoch, config, optimizer):
    if epoch >= config['optimizer_params']['step_epoch']:
        curr_lr = config['optimizer_params']['lr']
        changed_lr = curr_lr * (config['optimizer_params']['step_value'] ** (epoch-config['optimizer_params']['step_epoch']))
    else:
        changed_lr = config['optimizer_params']['lr']
    for g in optimizer.param_groups:
        g['lr'] = changed_lr
        c_lr = g['lr']
        print("Changed learning rate to {}".format(c_lr))

def train(config, rank):
    is_distributed = (rank >= 0)
    save_dir = Path(config['train_params']['save_dir'])
    weight_dir = save_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    batch_size = config['train_params']['batch_size']
    results_file = None 
    if rank in [0, -1]: results_file = open(save_dir / "results.txt", 'a')
    with open(save_dir / 'config.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)
    init_seeds(rank + config['train_params']['init_seed'])
    config['superglue_params']['GNN_layers'] = ['self', 'cross'] * config['superglue_params']['num_layers']

    loss_IS_alex = lpips.LPIPS(net = 'alex').to(device)
    loss_IS_vgg = lpips.LPIPS(net = 'vgg').to(device)

    RA_MMIR_model = RA_MMIR(config['RA_MMIR_params']).to(device)
    RAMM_Point_model = RAMM_Point(config['RAMM_Point_params']).to(device)

    RAMM_Point_model.eval()
    for _, k in RAMM_Point_model.named_parameters():
        k.requires_grad = False

    start_epoch = config['train_params']['start_epoch'] if config['train_params']['start_epoch'] > -1 else 0
    if config['superglue_params']['restore_path']:
        restore_dict = torch.load(config['superglue_params']['restore_path'], map_location = device)
        RA_MMIR_model.load_state_dict(clean_checkpoint(restore_dict['model'] if 'model' in restore_dict else restore_dict))
        print("Restored model weights..")
        if config['train_params']['start_epoch'] < 0:
            start_epoch = restore_dict['epoch'] + 1
    if is_distributed and config['train_params']['sync_bn']:
        RA_MMIR_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(RA_MMIR_model).to(device)
    pg0, pg1, pg2 = [], [], []
    for k, v in RA_MMIR_model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    # biases
        if hasattr(v, 'bin_score'):
            pg0.append(v.bin_score)
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.BatchNorm1d) or isinstance(v, nn.SyncBatchNorm):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if config['optimizer_params']['opt_type'].lower() == "adam":
        optimizer = optim.Adam(pg0, lr=config['optimizer_params']['lr'], betas=(0.9, 0.999))                    # adjust beta1 to momentum;
    else:
        optimizer = optim.SGD(pg0, lr=config['optimizer_params']['lr'], momentum=0.9, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': config['optimizer_params']['weight_decay']})
    optimizer.add_param_group({'params': pg2}) 
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    if config['superglue_params']['restore_path']:
        if ('optimizer' in restore_dict) and config['train_params']['restore_opt']:
            optimizer.load_state_dict(restore_dict['optimizer'])
            print("Restored optimizer...")
    ema = None
    if config['train_params']['use_ema']:
        ema = ModelEMA(RA_MMIR_model) if rank in [-1, 0] else None
        print("Keeping track of weights in ema..")
        if config['superglue_params']['restore_path']:
            if ('ema' in restore_dict) and (restore_dict['ema'] is not None):
                ema.ema.load_state_dict(restore_dict['ema'])
                ema.updates = restore_dict['ema_updates']
    if is_distributed:
        RA_MMIR_model = DDP(RA_MMIR_model, device_ids=[rank], output_device=rank)
    train_dataset = COCO_loader(config['dataset_params'], typ="train")
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train_params']['batch_size'],
                                            num_workers=config['train_params']['num_workers'],
                                            shuffle = False if is_distributed else True,
                                            sampler=sampler,
                                            collate_fn=collate_batch,
                                            pin_memory=True)
    num_batches = len(train_dataloader)
    if rank in [-1, 0]:
        val_dataset = COCO_valloader(config['dataset_params'])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                            num_workers=0,
                                            sampler=None,
                                            collate_fn=collate_batch,
                                            pin_memory=True)

    start_time = time.time()
    num_epochs = config['train_params']['num_epochs']
    best_val_score = 1e-10
    if rank in [-1, 0]: print("Started training for {} epochs".format(num_epochs))
    print("Number of batches: {}".format(num_batches))
    warmup_iters = config['optimizer_params']['warmup_epochs'] * num_batches
    change_lr(start_epoch, config, optimizer)

    for epoch in range(start_epoch, num_epochs):
        print("Started epoch: {} in rank {}".format(epoch + 1, rank))
        RA_MMIR_model.train()
        if rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(train_dataloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=num_batches)
        optimizer.zero_grad()
        mloss = torch.zeros(6, device=device)
        if rank in [-1, 0]: print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'Iteration', 'PosLoss', 'NegLoss', 'TotLoss', 'Dtime', 'Ptime', 'Mtime'))
        t5 = time_synchronized()
        for i, (orig_warped, warped, homographies) in pbar:
            ni = i + num_batches * epoch
            if ni < warmup_iters:
                xi = [0, warmup_iters]
                for _, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [0.0, config['optimizer_params']['lr']])
            t1 = time_synchronized()

            orig_warped_0 = orig_warped
            warped_0 = warped

            orig_warped = orig_warped.to(device, non_blocking=True)
            homographies = homographies.to(device, non_blocking=True)

            midpoint = len(orig_warped) // 2

            with torch.no_grad():
                all_match_index_0, all_match_index_1, all_match_index_2 = torch.empty(0, dtype=torch.int64, device=homographies.device), torch.empty(0, dtype=torch.int64, device=homographies.device), torch.empty(0, dtype=torch.int64, device=homographies.device)
                t2 = time_synchronized()

                superpoint_results = RAMM_Point_model.forward_train({'homography': homographies, 'image': orig_warped})
                keypoints = torch.stack(superpoint_results['keypoints'], 0)
                descriptors = torch.stack(superpoint_results['descriptors'], 0)
                scores = torch.stack(superpoint_results['scores'], 0)

                keypoints0, keypoints1 = keypoints[:midpoint, :, :], keypoints[midpoint:, :, :]
                descriptors0, descriptors1 = descriptors[:midpoint, :, :], descriptors[midpoint:, :, :]
                scores0, scores1 = scores[:midpoint, :], scores[midpoint:, :]
                images0, images1 = orig_warped[:midpoint, :, :, :], orig_warped[midpoint:, :, :, :]

                ma_0_lis = []
                ma_1_lis = []

                for k in range(midpoint):
                    ma_0, ma_1, miss_0, miss_1 = torch_find_matches(keypoints0[k], keypoints1[k], homographies[k], dist_thresh=3, n_iters=1)
                    ma_0_lis.append(ma_0)
                    ma_1_lis.append(ma_1)
                    all_match_index_0 = torch.cat([all_match_index_0, torch.empty(len(ma_0) + len(miss_0) + len(miss_1), dtype=torch.long, device=ma_0.device).fill_(k)])
                    all_match_index_1 = torch.cat([all_match_index_1, ma_0, miss_0, torch.empty(len(miss_1), dtype=torch.long, device=miss_1.device).fill_(-1)])
                    all_match_index_2 = torch.cat([all_match_index_2, ma_1, torch.empty(len(miss_0), dtype=torch.long, device=miss_0.device).fill_(-1), miss_1])

                if config['train_params']['debug'] and (i < config['train_params']['debug_iters']):
                    debug_image_plot(config['train_params']['debug_path'], keypoints0[k], keypoints1[k], ma_0, ma_1, images0[-1], images1[-1], epoch, i)

                match_indexes = torch.stack([all_match_index_0, all_match_index_1, all_match_index_2], -1)
                gt_vector = torch.ones(len(match_indexes), dtype=torch.float32, device=match_indexes.device)

            t3 = time_synchronized()

            superglue_input = {
                'keypoints0': keypoints0, 'keypoints1': keypoints1,
                'descriptors0': descriptors0, 'descriptors1': descriptors1,
                'image0': images0, 'image1': images1,
                'orig_warped': orig_warped_0, 'warped': warped_0,
                'ma_0': ma_0_lis, 'ma_1': ma_1_lis,
                'scores0': scores0, 'scores1': scores1,
                'matches': match_indexes,
                'gt_vec': gt_vector
            }

            total_loss, pos_loss, neg_loss, tps_target_imgs, warped_imgs, scorce_imgs = RA_MMIR_model(superglue_input, **{'mode': 'train'})

            loss_SI_LPIPS = []
            # loss_SI_SISM = []
            for i in range(0, batch_size):
                tps_img = torch.from_numpy(tps_target_imgs[i]).to(device)
                warped_img = torch.from_numpy(warped_imgs[i]).to(device)

                tps_img = tps_img.transpose(2, 0)
                warped_img = warped_img.transpose(2, 0)

                tps_img = tps_img.transpose(2, 1)
                warped_img = warped_img.transpose(2, 1)

                tps_img = tps_img.unsqueeze(0)
                warped_img = warped_img.unsqueeze(0)

                loss = float(loss_IS_alex(tps_img, warped_img))
                loss_SI_LPIPS.append(loss)

            loss_SI_LPIPS_batch = np.array(loss_SI_LPIPS).mean()

            print(total_loss, pos_loss, neg_loss, loss_SI_LPIPS_batch)

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t4 = time_synchronized()

            if ema:
                ema.update(RA_MMIR_model)
            data_time, preprocess_time, model_time = torch.tensor(t1 - t5, device=device), torch.tensor(t3-t2, device=device), torch.tensor(t4-t3, device=device)
            loss_items = torch.stack((pos_loss, neg_loss, total_loss, data_time, preprocess_time, model_time)).detach()

            if is_distributed:
                loss_items = reduce_tensor(loss_items)
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s' * 2 + '%10.4g' * 7) % (str(epoch), mem, i, *mloss)
                pbar.set_description(s)
                if ((i+1) % config['train_params']['log_interval']) == 0:
                    write_str = "Epoch: {} Iter: {}, Loss: {}\n".format(epoch, i, mloss[0].item())
                    results_file.write(write_str)
                if ((i+1) % 2000) == 0:
                    ckpt = {'epoch': epoch,
                            'iter': i,
                            'ema': ema.ema.state_dict() if ema else None,
                            'ema_updates': ema.updates if ema else 0,
                            'model': RA_MMIR_model.module.state_dict() if is_distributed else RA_MMIR_model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                    torch.save(ckpt, weight_dir / 'att_emau_coco_feature.pt')
                    if use_wandb:
                        wandb.save(str(weight_dir / 'att_emau_coco_feature.pt'))
                t5 = time_synchronized()
        if rank in [-1, 0]:
            print("\nDoing evaluation..")
            with torch.no_grad():
                if ema:
                    eval_superglue = ema.ema
                else:
                    eval_superglue = RA_MMIR_model.module if is_distributed else RA_MMIR_model
                results = test_model(val_dataloader, RAMM_Point_model, eval_superglue, config['train_params']['val_images_count'], device)
            ckpt = {'epoch': epoch,
                    'iter': -1,
                    'ema': ema.ema.state_dict() if ema else None,
                    'ema_updates': ema.updates if ema else 0,
                    'model': RA_MMIR_model.module.state_dict() if is_distributed else RA_MMIR_model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'metrics': results}
            torch.save(ckpt, weight_dir / 'last_att_emau_coco_feature.pt')
            if use_wandb:
                wandb.save(str(weight_dir / 'last_att_emau_coco_feature.pt'))
                results_file.flush()
                wandb.save(str(save_dir / "results_emau_coco_feature.txt"))
            if results['weight_score'] > best_val_score:
                best_val_score = results['weight_score']
                print("Saving best model at epoch {} with score {}".format(epoch, best_val_score))
                torch.save(ckpt, weight_dir / 'best_att_emau_coco_feature.pt')
                if use_wandb:
                    wandb.save(str(weight_dir / 'best_att_emau_coco_feature.pt'))
        change_lr(epoch, config, optimizer)
    if rank > 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/coco_config.yaml", help="Path to the config file")
    parser.add_argument('--local_rank', type=int, default=-1, help="Rank of the process incase of DDP")
    opt = parser.parse_args()
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if opt.local_rank >= 0:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda:1', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        if "cpu" not in device:
            torch.cuda.set_device(device)
    with open(opt.config_path, 'r') as file:
        config = yaml.full_load(file)
    config["train_params"]['save_dir'] = increment_path(Path(config['train_params']['output_dir']) / config['train_params']['experiment_name'])
    if opt.local_rank in [0, -1]:
        for i, k in config.items():
            print("{}: ".format(i))
            print(k)
    
    download_base_files()
    use_wandb = False
    if config['train_params']['use_wandb']:
        import wandb
        wandb.init(name=config['train_params']['experiment_tag'], config=config, notes="train", project="superglue")
        use_wandb = True
    train(config, opt.local_rank)
