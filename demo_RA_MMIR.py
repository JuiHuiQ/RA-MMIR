from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import os
from models.matching import Matching
from Radiation_transformation import test
import numpy as np
from utils.common import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor, download_base_files, weights_mapping)

from TLGAN.fusion.RGB_run import att_run
from ATGAN.test_gray import fusion_test
from TIMGAN.test import test_fuison
from tps_stn.single_visualize_kpts import TPS
from collections import deque
import time

torch.set_grad_enabled(False)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class KS:
    def __init__(self):
        self.kpts0 = 0
        self.kpts1 = 0
        self.sources = 1

def sort(a):
    "“”排序返回索引"""
    sorted = list(np.sort(a))
    indices = list(np.argsort(sorted))

    ser = np.zeros_like(a)
    for i in range(len(a)):
        ser[i] = indices[sorted.index(a[i])]
        indices.pop(sorted.index(a[i]))
        sorted.remove(a[i])

    return ser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_1', type=str, default='0',                     # IP:0
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')

    parser.add_argument(
        '--input_2', type=str, default='1',                     # IP:1
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')

    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')

    parser.add_argument(
        '--skip', type=int, default = 1,
        help='Images to skip if input is a movie or directory')

    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--RA_MMIR', default='outdoor',
        help='SuperGlue weights')

    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')

    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.05,
        help='SuperPoint keypoint detector confidence threshold')

    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')

    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')

    parser.add_argument(
        '--match_threshold', type=float, default=0.80,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')

    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')

    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    download_base_files()

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    try:
        curr_weights_path = str(weights_mapping[opt.superglue])
    except:
        if os.path.isfile(opt.superglue) and (os.path.splitext(opt.superglue)[-1] in ['.pt', '.pth']):
            curr_weights_path = str(opt.superglue)
        else:
            raise ValueError("Given --superglue path doesn't exist or invalid")

    config = {
        'RAMM_Point': {
            'name': 'RAMM_Point',
            'using_bn': False,
            'grid_size': 8,
            'pretrained_model': 'none',
            'backbone': {
                'backbone_type': 'VGG',
                'vgg': {
                    'channels': [64, 64, 64, 64, 128, 128, 128, 128], }, },
            'det_head': {
                'feat_in_dim': 128},
            'des_head': {                                                                       # descriptor head
                'feat_in_dim': 128 ,
                'feat_out_dim': 256, },
            'det_thresh': 0.001,                                                                # 1/65
            'nms': 4,
            'topk': -1,
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'RA_MMIR': {
            'weights_path': curr_weights_path,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    keys = ['keypoints', 'scores', 'descriptors']

    vs_1 = VideoStreamer(opt.input_1, opt.resize, opt.skip, opt.image_glob, opt.max_length)
    vs_2 = VideoStreamer(opt.input_2, opt.resize, opt.skip, opt.image_glob, opt.max_length)

    frame_1, frame_1_color, ret_1 = vs_1.next_frame()
    frame_2, frame_2_color, ret_2 = vs_2.next_frame()
    frame_3, ret_3 = vs_1.next_frame_YCBCR()

    matching = Matching(config, frame_1).eval().to(device)

    assert ret_1, 'Error when reading the first frame (try different --input_1?)'
    assert ret_2, 'Error when reading the first frame (try different --input_2?)'
    assert ret_3, 'Error when reading the first frame (try different --input_1_RGB?)'

    frame_tensor_1 = frame2tensor(frame_1, device)
    frame_tensor_2 = frame2tensor(frame_2, device)
    frame_tensor_3 = frame2tensor(frame_3, device)

    last_data_1 = matching.RAMM_Point_bn({'image': frame_tensor_1}, 1)
    last_data_1 = {k+'0': last_data_1[k] for k in keys}
    last_data_1['image0'] = frame_tensor_1

    last_data_2 = matching.RAMM_Point_bn({'image': frame_tensor_2}, 0)
    last_data_2 = {k+'0': last_data_2[k] for k in keys}
    last_data_2['image0'] = frame_tensor_2


    last_frame_1 = frame_1
    last_frame_2 = frame_2

    last_image_id = 0

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', (640 * 2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    class FixedSizeQueue:
        def __init__(self, max_size):
            self.queue = deque(maxlen=max_size)

        def enqueue(self, item):
            self.queue.append(item)

        def dequeue(self):
            return self.queue.popleft()

        def is_empty(self):
            return len(self.queue) == 0

        def is_full(self):
            return len(self.queue) == self.queue.maxlen

        def size(self):
            return len(self.queue)

    ks_list = []
    T_list = FixedSizeQueue(max_size = 40)
    while True:
        frame_1, frame_1_color, ret_1 = vs_1.next_frame()
        frame_2, frame_2_color, ret_2 = vs_2.next_frame()
        frame_3, ret_3 = vs_1.next_frame_YCBCR()

        if not ret_1:
            print('Finished demo_superglue.py')
            break

        if not ret_2:
            print('Finished demo_superglue.py')
            break

        stem0, stem1 = vs_1.i - 1, vs_2.i - 1

        frame_tensor_1 = frame2tensor(frame_1, device)
        frame_tensor_2 = frame2tensor(frame_2, device)
        frame_tensor_3 = frame2tensor(frame_3, device)

        start_time = time.time()

        pred = matching({'image0': frame_tensor_1, 'image1': frame_tensor_2})

        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()

        matches = pred['matches0'][0].cpu().numpy()                                         # 匹配关系
        confidence = pred['matching_scores0'][0].cpu().numpy()                              # 匹配置信度数组
        timer.update('forward')

        valid = matches > -1                                                                # 特征点对应关系
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]                                                      # 根据匹配置信度排序匹配特征点

        color = cm.jet(confidence[valid])

        mkpts0_0 = np.zeros_like(mkpts0)
        mkpts1_0 = np.zeros_like(mkpts0)

        sort_match = sort(confidence[valid])

        text = [
            'RA_MMIR',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]

        """选择superpoint/superpint_bn"""
        k_thresh = matching.RAMM_Point_bn.config['keypoint_threshold']

        m_thresh = matching.RA_MMIR_bn.config['match_threshold']

        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]

        # out, out_2, out_3, out_4 = make_matching_plot_fast(
        #     frame_1_color, frame_2_color, kpts0, kpts1, mkpts0, mkpts1, color, text,
        #     path=None, show_keypoints=opt.show_keypoints, small_text=small_text)                        # 得到匹配图像

        """求变换矩阵"""
        if mkpts0.shape[0] >= 10:
            M, err = test(mkpts0_0, mkpts1_0)                                                           # 拟合全局特征对应变换
            err = 100
            # if err <= 200:
            target = np.zeros_like(mkpts0)
            mkpts0_H = []
            target_H = []
            rows, cols = frame_1.shape[:2]
            xishu = 0.5
            # if M:
            for i in range(mkpts0_0.shape[0]):
                target[i] = M.transform(mkpts0_0[i])                                                 # mkpts0 --> target
            for j in [0, 3, 6, 9]:
                mkpts0_H.append(mkpts0_0[j])
                target_H.append(target[j])

            mkpts0_H = np.float32(mkpts0_H)
            target_H = np.float32(target_H)
            mkpts1_0 = np.float32(mkpts1_0)

            H = cv2.getPerspectiveTransform(mkpts0_H, target_H)
            frame_4 = np.ascontiguousarray(cv2.warpPerspective(frame_1, H, (cols, rows)))
            frame_4_color = np.ascontiguousarray(cv2.warpPerspective(frame_1_color, H, (cols, rows)))
            if err < 200:
                # frame_4 = np.ascontiguousarray(cv2.warpPerspective(frame_3, H, (cols, rows)))
                # source_image = cv2.cvtColor(frame_4, cv2.COLOR_GRAY2RGB)
                # print(frame_4.shape)
                frame_4_color = TPS.forward(None, frame_4_color, target, mkpts1_0, 0)                                   # tps
                end_time = time.time()
                time_ = end_time - start_time
                print(time_)

                frame_4_color = np.ascontiguousarray(frame_4_color)

                # frame_5 = frame_4[:, :, 0]
                # fusion = att_run(frame_3, frame_2)                                                # TLGAN
                fusion = fusion_test(frame_4, frame_2, frame_4_color)                               # ATGAN
                # fusion = test_fuison(frame_3, frame_2)                                            # TIMGAN

                # fusion = fusion.transpose([2, 3, 0, 1])
                # fusion = np.squeeze(fusion, axis=0)
                # fusion = np.squeeze(fusion, axis=1)
                # fusion = fusion.detach().cpu().numpy()
                # fusion = fusion.astype(np.uint8)
                # fusion = np.clip(fusion, 0, 255)

                # obj = detect(fusion)

                # print(fusion.shape)
                # cv2.imshow('transform', fusion)
            else:
                # source_image = cv2.cvtColor(frame_4, cv2.COLOR_GRAY2RGB)
                # frame_4 = TPS.forward(None, source_image, target, mkpts1_0)  # tps变换
                # frame_4 = np.ascontiguousarray(frame_4)
                # frame_4 = frame_4[:, :, 0]

                target_img = pred['target']
                # target_img = np.ascontiguousarray(target_img)
                target_img = target_img[:, :, 0]

                # fusion = att_run(frame_3, frame_2)                                                # TLGAN
                fusion = all(target_img, frame_2)                                                   # ATGAN
                # fusion = test_fuison(frame_3, frame_2)                                            # TIMGAN

                # fusion = fusion.transpose([2, 3, 0, 1])
                # fusion = np.squeeze(fusion, axis=0)
                # fusion = np.squeeze(fusion, axis=1)
                # fusion = fusion.detach().cpu().numpy()
                # fusion = fusion.astype(np.uint8)
                # fusion = np.clip(fusion, 0, 255)

                # obj = detect(fusion)
                cv2.imshow('transform', fusion)


        if not opt.no_display:
            # cv2.imshow('RA_MMIR', out)
            key = chr(cv2.waitKey(1) & 0xFF)

            if key == 'q':
                vs_1.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':      # set the current frame as anchor;
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor_1
                last_frame = frame_1
                last_image_id = (vs_1.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress;
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            # stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            # cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs_1.cleanup()
    vs_2.cleanup()
