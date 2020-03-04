# -*- coding: utf-8 -*-
"""
bn与权值初始化对比：
我们发现在relu激活的情况下，仅有kaiming初始化权值在不同层的std会在0.4-1.0之间波动
而加上bn层取消kaiming初始化，权值在所有层都非常稳定，std全在0.57-0.58之间
"""
import torch
import torch.nn as nn
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

            print("layers:{}, std:{}".format(i, x.std().item()))

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                # method 1
                # nn.init.normal_(m.weight.data, std=1)    # normal: mean=0, std=1

                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)


neural_nums = 256
layer_nums = 100
batch_size = 16

net = MLP(neural_nums, layer_nums)
#net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)



















