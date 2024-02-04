import torch
import numpy as np

pt = torch.load('L:\\CODE\\matching\\SuperGlue\\SuperGlue_training-main\\tps_stn\\mnist_data\\MNIST\\processed\\training.pt')
print(pt)

# r1 = 0.9
# x = np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (4 - 1))
#
# print(x)