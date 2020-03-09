"""
模型的加载，加载网络或者仅加载参数
"""
import torch
import torch.nn as nn
from tools.common_tools import set_seed
set_seed(2020)

class LeNet2(nn.Module):
    def __init__(self, classes):
        super(LeNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def initialize(self):
        for p in self.parameters():
            p.data.fill_(20200309)


# ================================== load net ===========================
#flag = 1
flag = 0
if flag:

    path_model = "./model.pkl"
    net_load = torch.load(path_model) # 直接将整个网络定义以及参数加载进来

    print(net_load)

# ================================== load state_dict ===========================
#flag = 1
flag = 0
if flag:

    path_state_dict = "./model_state_dict.pkl"
    state_dict_load = torch.load(path_state_dict) # 加载网络参数

    print(state_dict_load.keys()) # 打印参数组的名称，例如 XXX.weight / bias

# ================================== update state_dict ===========================
flag = 1
#flag = 0
if flag:

    net_new = LeNet2(classes=2020) # 加载参数的话，需要先实例化网络

    print("加载前: ", net_new.features[0].weight[0, ...])    # 打印第一层的权重参数
    net_new.load_state_dict(state_dict_load)                # 将参数装载到网络中
    print("加载后: ", net_new.features[0].weight[0, ...])





