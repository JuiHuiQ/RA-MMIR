import logging
import numpy as np
from torch import Tensor

# visdom可视化参数定义
env = "default"  	# visdom 环境
vis_port = 8097 	# visdom 端口

# Data settings
DATA_ROOT = '.\\datalist'    # 测试图片地址
MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
STD = Tensor(np.array([0.229, 0.224, 0.225]))
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
CROP_SIZE = 513
IGNORE_LABEL = 255

# Model definition 模型定义
N_CLASSES = 32     # 分类数为20，原21(final.pth)
N_LAYERS = 101     # 使用ResNet-101
STRIDE = 8         # 步距；步幅
BN_MOM = 3e-4
EM_MOM = 0.9
STAGE_NUM = 3      # 阶段数

# Training settings 训练定义
BATCH_SIZE = 1     # 批量大小，原16
ITER_MAX = 30000   # 最大迭代次数
ITER_SAVE = 2000   # 迭代器保存

LR_DECAY = 10      # 学习率衰变
LR = 9e-3          # 学习率，原9e-2
LR_MOM = 0.9       # 学习冲量
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

DEVICE = 0          # 选择本机GPU
DEVICES = [0]       # list(range(0, 4))，训练时用GPU编号，单机只有0号

LOG_DIR = '.\\logdir'    # log目录
MODEL_DIR = '.\\models'  # model目录
NUM_WORKERS = 4          # 物理内核数，线程数; 原16

logger = logging.getLogger('trainaug')                                      # 返回具有指定名称的记录器，必要时创建它
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)                                                  # 设置格式化程序
logger.addHandler(ch)
