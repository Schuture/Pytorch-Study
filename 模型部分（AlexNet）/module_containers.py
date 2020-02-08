import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


# ============================ Sequential
class LeNetSequential(nn.Module): # 直接按顺序将层加入来构建sequential
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1) # 将feature map转换为向量
        x = self.classifier(x)
        return x


class LeNetSequentialOrderDict(nn.Module): # 使用有序字典来构建sequential
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# =============================================================================
# net = LeNetSequential(classes=2)
# # net = LeNetSequentialOrderDict(classes=2)
# 
# fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)
# 
# output = net(fake_img)
# 
# print(net)
# 
# print(output)
# =============================================================================


# ============================ ModuleList

class ModuleList(nn.Module): # 使用ModuleList来构建有重复模块的网络
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x


# =============================================================================
# net = ModuleList()
# 
# print(net)
# 
# fake_data = torch.ones((10, 10)) # 一个batch10个数据，每个数据为10维向量
# 
# output = net(fake_data)
# 
# print(output)
# =============================================================================


# ============================ ModuleDict

class ModuleDict(nn.Module): # 使用ModuleDict来构建某些层可选的网络
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


# =============================================================================
# net = ModuleDict()
# 
# fake_img = torch.randn((4, 10, 32, 32))
# 
# output = net(fake_img, 'conv', 'relu')
# 
# print(output)
# =============================================================================




# 4 AlexNet，可以自行查看torchvision中的定义代码

alexnet = torchvision.models.AlexNet()

print(alexnet)



