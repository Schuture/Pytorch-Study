"""
模型的保存，使用保存整个模型、保存参数两种方式
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


net = LeNet2(classes=2020)

# "训练"
print("训练前: ", net.features[0].weight[0, ...])
net.initialize()
print("训练后: ", net.features[0].weight[0, ...])

path_model = "./model.pkl"
path_state_dict = "./model_state_dict.pkl"

# 保存整个模型
torch.save(net, path_model)

# 保存模型参数
net_state_dict = net.state_dict()
torch.save(net_state_dict, path_state_dict)







