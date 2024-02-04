import torch
from .modules.cnn.vgg_att_backbone import VGGBackbone_att
from torch.nn import DataParallel
from .modules.cnn.cnn_heads import DetectorHead, DescriptorHead
from pathlib import Path
import settings
import torch.nn.functional as F

def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_w = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (width - border))
    mask_h = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (height - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

def remove_borders_projected(keypoints, scores, border, height, width, homo_matrix):
    projected_keypoints = warp_keypoints(keypoints, torch.inverse(homo_matrix))
    mask_w = (projected_keypoints[:, 0] >= border) & (projected_keypoints[:, 0] < (width - border))
    mask_h = (projected_keypoints[:, 1] >= border) & (projected_keypoints[:, 1] < (height - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

def warp_keypoints(keypoints, homography_mat):
    source = torch.cat([keypoints, torch.ones(len(keypoints), 1).to(keypoints.device)], dim=-1)
    dest = (homography_mat @ source.T).T
    # dest /= dest[:, 2:3]
    dest_c = dest[:, 2:3]
    dest_b = dest_c.clone()
    dest /= dest_b
    return dest[:, :2]

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations;
        在关键点位置插入描述符;
    """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1                                                                                       # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

class RAMM_Point(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network.
        SuperPoint网络的Pytorch定义。
    """

    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config, input_channel=1, grid_size=8, device='gpu', using_bn=True):
        super(RAMM_Point, self).__init__()
        self.config = {**self.default_config, **config}
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']

        # self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)

        self.backbone = VGGBackbone_att(config['backbone']['vgg'], settings.N_CLASSES, input_channel, device=device)
        self.backbone = DataParallel(self.backbone, device_ids=[settings.DEVICE])  # 使用多GPU

        self.detector_head = DetectorHead(self.config, input_channel=config['det_head']['feat_in_dim'],
                                          grid_size=grid_size, using_bn=using_bn)

        self.descriptor_head = DescriptorHead(input_channel=config['des_head']['feat_in_dim'],
                                              output_channel=config['des_head']['feat_out_dim'],
                                              grid_size=grid_size, using_bn=using_bn)


        path = Path(__file__).parent / '.\\weights\\superpoint_coco_emau_feature_3090.pth'

        self.load_state_dict(torch.load(str(path)))

        mk = config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded RAMM_Point model')

    def forward(self, x, vi_or_ir, curr_max_kp=None, curr_key_thresh=None):
        """ Forward pass that jointly computes unprocessed point and descriptor
            tensors.
            Input
              x: Image pytorch tensor shaped N x 1 x H x W.
            Output
              semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
              desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        if curr_max_kp is None:
            curr_max_kp = self.config['max_keypoints']
        if curr_key_thresh is None:
            curr_key_thresh = self.config['keypoint_threshold']

        if isinstance(x, dict):
            feat_map, att_class = self.backbone(x['image'])
        else:
            feat_map, att_class = self.backbone(x)

        pred = att_class.max(dim=1)[1]
        pred = pred.permute(1, 2, 0)
        w, h, c = pred.size()
        pred = pred.expand(w, h, 1)
        pred_mask = pred.permute(2, 0, 1)

        pred = pred_mask.unsqueeze(0)

        pred = torch.tensor(pred, dtype=torch.float32)
        pred = F.interpolate(pred, size=(60, 80), mode='bilinear')

        pred = pred.repeat_interleave(128, dim=1)

        RAMM_Gauge_Field = 1.0 * pred + 1.0 * feat_map                                                                  # RAMM_Gauge_Field

        RAMM_Gauge_Field = F.interpolate(RAMM_Gauge_Field, size=(480, 640), mode='bilinear', align_corners=True)

        if vi_or_ir == 1 :
            keypoints, scores, h, w = self.detector_head(RAMM_Gauge_Field, curr_max_kp, curr_key_thresh)
            desc_outputs = self.descriptor_head(feat_map)
        else:
            keypoints, scores, h, w = self.detector_head(RAMM_Gauge_Field, curr_max_kp, curr_key_thresh)
            desc_outputs = self.descriptor_head(feat_map)

        # Extract descriptors;
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, desc_outputs)]


        return {
                'keypoints': keypoints,
                'scores': scores,
                'descriptors': descriptors
                }

    def forward_train(self, data):
        """
        Compute keypoints, scores, descriptors for image during training.
        Here the first half contains the original image, the next half contains warped images and its
        keypoints should be removed from border according to the given homography matrix
        """

        # Shared Encoder;
        homo_matrices = data['homography']                                                                              # if batch size of image is N, the batch size of homography is N/2

        if isinstance(data, dict):
            feat_map, att_class = self.backbone(data['image'])
        else:
            feat_map, att_class = self.backbone(data)

        """RAMM_Gauge_Field"""
        pred = att_class.max(dim=1)[1]
        pred = pred.permute(1, 2, 0)
        w, h, c = pred.size()
        pred = pred.expand(w, h, 1)
        pred_mask = pred.permute(2, 0, 1)

        pred = pred_mask.unsqueeze(0)

        pred = torch.tensor(pred, dtype=torch.float32)
        pred = F.interpolate(pred, size=(60, 80), mode='bilinear')

        pred = pred.repeat_interleave(128, dim=1)

        RAMM_Gauge_Field = 1.0 * pred + 1.0 * feat_map  # RAMM_Gauge_Field

        RAMM_Gauge_Field = F.interpolate(RAMM_Gauge_Field, size=(480, 640), mode='bilinear', align_corners=True)

        keypoints, scores, h, w = self.detector_head(RAMM_Gauge_Field, self.config['max_keypoints'], self.config['keypoint_threshold'])

        # Discard keypoints near the image borders; 丢弃图像边界附近的关键点;
        homo_mat_index = 0
        results = []
        mid_point = len(keypoints) // 2
        for i, (k, s) in enumerate(zip(keypoints, scores)):
            if i < mid_point:                                                                                           # orig image;
                results.append(remove_borders(k, s, self.config['remove_borders'], h*8, w*8))
            else:
                homo_matrix = homo_matrices[homo_mat_index]
                homo_mat_index += 1
                results.append(remove_borders_projected(k, s, self.config['remove_borders'], h * 8, w * 8, homo_matrix))
        keypoints, scores = list(zip(*results))
        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        keypoints, scores = list(keypoints), list(scores)

        for i, (k, s) in enumerate(zip(keypoints, scores)):
            """
            Occurence of below condition is very rare as we are sampling keypoints above threshold of 0 itself and then
            sampling max_keypoints from it. But incase if it happends then we are randomly adding some pixel locations to 
            without checking any conditions with respect to preexisting keypoints.
            """
            if len(k) < self.config['max_keypoints']:
                print("Rare condition executed")
                to_add_points = self.config['max_keypoints'] - len(k)
                random_keypoints = torch.stack([torch.randint(0, w*8, (to_add_points,), dtype=torch.float32, device=k.device), \
                                                torch.randint(0, h*8, (to_add_points,), dtype=torch.float32, device=k.device)], 1)
                keypoints[i] = torch.cat([keypoints[i], random_keypoints], dim=0)
                scores[i] = torch.cat([scores[i], torch.ones(to_add_points, dtype=torch.float32, device=s.device)*0.1], dim=0)

        # Compute the dense descriptors;
        desc_outputs = self.descriptor_head(feat_map)

        # Extract descriptors;
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, desc_outputs)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            }
