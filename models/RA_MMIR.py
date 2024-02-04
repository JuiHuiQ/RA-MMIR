from copy import deepcopy
import torch
from torch import nn
from tps_stn.single_visualize_kpts import TPS
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
from .ts_scatter import scatter_mean

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class LayerNorm(nn.Module):
    """Construct a layer-norm module (See citation for details). """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps                                                                                      # 1e-6

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        std = x.std(-2, keepdim=True)
        return torch.reshape(self.a_2, (1, -1, 1)) * ((x - mean) / (std + self.eps)) + torch.reshape(self.b_2, (1, -1, 1))

def MLP(channels: list, use_layernorm, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if use_layernorm:
                layers.append(LayerNorm(channels[i]))
            elif do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers, use_layernorm=False):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim], use_layernorm=use_layernorm)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy"""
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, use_layernorm=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim], use_layernorm=use_layernorm)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, use_layernorm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4, use_layernorm=use_layernorm)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm                                                                                         # multiply probabilities by M+N

    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1                                                        # traceable in 1.1


class RA_MMIR(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights_path': None,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'use_layernorm': False
    }

    def __init__(self, config, frame_1 = None):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'], use_layernorm=self.config['use_layernorm'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'], use_layernorm=self.config['use_layernorm'])
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(self.config['bin_value'] if 'bin_value' in self.config else 1.0))

        self.register_parameter('bin_score', bin_score)
        self.frame_1 = frame_1
        self.tps_st = TPS()                                                                 # tps_stn

        if self.config['weights_path']:
            weights = torch.load(self.config['weights_path'], map_location="cpu")
            if ('ema' in weights) and (weights['ema'] is not None):
                load_dict = weights['ema']
            elif 'model' in weights:
                load_dict = weights['model']
            else:
                load_dict = weights
            self.load_state_dict(load_dict)
            print('Loaded SuperGlue model (\"{}\" weights)'.format(self.config['weights_path']))

    def forward(self, data, **kwargs):
        """
        Run SuperGlue on a pair of keypoints and descriptors;
        在一对关键点和描述符上运行SuperGlue;
        """

        if kwargs.get('mode', 'test') == "train":
            return self.forward_train(data)

        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0_0, kpts1_0 = data['keypoints0'], data['keypoints1']

        if kpts0_0.shape[1] == 0 or kpts1_0.shape[1] == 0:                              # no keypoints
            shape0, shape1 = kpts0_0.shape[:-1], kpts1_0.shape[:-1]
            return {
                'matches0': kpts0_0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1_0.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0_0.new_zeros(shape0),
                'matching_scores1': kpts1_0.new_zeros(shape1),
                }

        # Keypoint normalization
        kpts0 = normalize_keypoints(kpts0_0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1_0, data['image1'].shape)

        # Keypoint MLP encoder
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport
        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])        # sinkhorn

        # Get the matches with score above "match_threshold"
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)

        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)

        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        matches = indices0[0].cpu().numpy()

        valid = matches > -1
        kpts0_0_0 = kpts0_0[0].cpu().numpy()
        kpts1_0_0 = kpts1_0[0].cpu().numpy()
        mkpts0 = kpts0_0_0[valid]
        mkpts1 = kpts1_0_0[matches[valid]]


        return {
            'matches0': indices0,                                                   # use -1 for invalid match
            'matches1': indices1,                                                   # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,

            # 'valid' : valid,
            # 'kpts0': kpts0_0_0,
            # 'kpts1': kpts1_0_0,
            # 'mkpts0': mkpts0,
            # 'mkpts1': mkpts1,

            # 'target': target
            }

    def forward_train(self, data):
        """Run RA-MMIR on a pair of keypoints and descriptors"""
        batch_size = data['image0'].shape[0]
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0_0, kpts1_0 = data['keypoints0'], data['keypoints1']

        kpts0 = normalize_keypoints(kpts0_0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1_0, data['image1'].shape)

        # Keypoint MLP encoder
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport
        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold"
        gt_indexes = data['matches']

        neg_flag = (gt_indexes[:, 1] == -1) | (gt_indexes[:, 2] == -1)                                          # valid

        source_image = data["orig_warped"]
        warped_image = data["warped"]
        source_image_np = source_image.numpy()
        warped_image_np = warped_image.numpy()

        source = []
        tps_target = []
        warped = []
        for i in range(0, int(batch_size)):
            kpts0_0_0 = kpts0_0[i].detach().cpu().numpy()                                                               # max_keypoints
            kpts1_0_0 = kpts1_0[i].detach().cpu().numpy()

            ma_0 = data['ma_0'][i].detach().cpu().numpy()
            ma_1 = data['ma_1'][i].detach().cpu().numpy()

            mkpts0 = []
            mkpts1 = []

            for key0, key1 in zip(ma_0, ma_1):
                mkpts0.append(kpts0_0_0[key0])
                mkpts1.append(kpts1_0_0[key1])

            source_image_c = source_image_np[i][0]
            source_image_channel = source_image_np[i][0] * 255.0
            warped_image_channel = warped_image_np[i][0]

            source_image_c = cv2.cvtColor(source_image_c, cv2.COLOR_GRAY2BGR)
            source_image_channel = cv2.cvtColor(source_image_channel, cv2.COLOR_GRAY2BGR)
            warped_image_channel = cv2.cvtColor(warped_image_channel, cv2.COLOR_GRAY2BGR)
            source_img = np.ascontiguousarray(source_image_c)
            warped_img = np.ascontiguousarray(warped_image_channel)

            target = self.tps_st.forward(source_image_channel, mkpts0, mkpts1, batch_size)
            # target_img = np.ascontiguousarray(target)

            source.append(source_img)
            tps_target.append(target)
            warped.append(warped_img)


        loss_pre_components = scores[gt_indexes[:, 0], gt_indexes[:, 1], gt_indexes[:, 2]]
        loss_pre_components = torch.clamp(loss_pre_components, min = -100, max = 0.0)
        loss_vector = -1 * loss_pre_components
        neg_index, pos_index = gt_indexes[:, 0][neg_flag], gt_indexes[:, 0][~neg_flag]
        batched_pos_loss, batched_neg_loss = scatter_mean(loss_vector[~neg_flag], pos_index, dim_size=batch_size), scatter_mean(loss_vector[neg_flag], neg_index, dim_size=batch_size)
        pos_loss, neg_loss = self.config['pos_loss_weight']*batched_pos_loss.mean(), self.config['neg_loss_weight']*batched_neg_loss.mean()       # 计算损失
        loss = pos_loss + neg_loss

        return loss, pos_loss, neg_loss, tps_target, warped, source