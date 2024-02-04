import logging
import numpy as np
from torch import Tensor

env = "default"  	# visdom
vis_port = 8097 	# visdom

# Data settings
DATA_ROOT = 'L:\\dataset\\VOC2012_\\VOC2012_att\\'
MEAN = Tensor(np.array([0.449]))
STD = Tensor(np.array([0.226]))

SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
CROP_SIZE = [513, 513]
IGNORE_LABEL = 255

# Model definition
N_CLASSES = 21
N_LAYERS = 50
STRIDE = 5
BN_MOM = 3e-4

EM_MOM = 0.9
TH_MOM = 0.005

STAGE_NUM = 2

# Training settings
BATCH_SIZE = 2
ITER_MAX = 2000000
ITER_SAVE = 2000

LR_DECAY = 32
LR = 9e-6
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

DEVICE = 0
DEVICES = [0]

LOG_DIR = '.\\logdir'
MODEL_DIR = '.\\models'
NUM_WORKERS = 2

logger = logging.getLogger('trainaug')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
