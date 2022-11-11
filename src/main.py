import torch
import numpy as np

torch.set_printoptions(threshold=np.inf)

input_n = 1
input_c = 4
input_d = 4
input_h = 4
input_w = 4

kernel_d = 3
kernel_h = 3
kernel_w = 3

stride_d = 1
stride_h = 1
stride_w = 1

x = torch.randn(input_n, input_c, input_d, input_h, input_w)

conv3d = torch.nn.Conv3d(input_c, 4, kernel_size=(kernel_d, kernel_h, kernel_w), stride=(stride_d, stride_h, stride_w), padding=0)
y = conv3d(x)

print("filter shape: ", conv3d.weight.shape)
print("y.shape: ", y.shape)