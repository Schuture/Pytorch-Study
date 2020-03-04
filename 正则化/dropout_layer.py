# -*- coding:utf-8 -*-
"""
dropout 使用实验，看看dropout层如何添加，训练与测试有什么异同
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from tools.common_tools import set_seed
from torch.utils.tensorboard import SummaryWriter

# set_seed(1)  # 设置随机种子


class Net(nn.Module):
    def __init__(self, neural_num, d_prob=0.5):
        super(Net, self).__init__()

        self.linears = nn.Sequential(

            nn.Dropout(d_prob), # Dropout层记住放在线性层之前
            nn.Linear(neural_num, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linears(x)

input_num = 10000
x = torch.ones((input_num, ), dtype=torch.float32) # 初始化一个全1向量作为输入

net = Net(input_num, d_prob=0.5)
net.linears[1].weight.detach().fill_(1.)

net.train() # 训练模式会随机失活
y = net(x)
print("output in training mode", y)

net.eval() # 测试模式不会随机失活
y = net(x)
print("output in eval mode", y)

















