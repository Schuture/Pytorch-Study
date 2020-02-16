import os
import torch
import random
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module): # 多层感知机
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x

    def initialize(self): # 在这里选择不同的初始化方式
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))    # normal: mean=0, std=1

                # a = np.sqrt(6 / (self.neural_num + self.neural_num))
                #
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                #
                # nn.init.uniform_(m.weight.data, -a, a)

                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))
                nn.init.kaiming_normal_(m.weight.data)

flag = 0
# flag = 1

if flag:
    layer_nums = 100 # 一百层的全连接神经网络
    neural_nums = 256
    batch_size = 16

    net = MLP(neural_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

    output = net(inputs)
    print(output)

# ======================================= calculate gain =======================================

# flag = 0
flag = 1

if flag:

    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print('gain:{}'.format(gain))

    tanh_gain = nn.init.calculate_gain('tanh')
    print('tanh_gain in PyTorch:', tanh_gain)


























