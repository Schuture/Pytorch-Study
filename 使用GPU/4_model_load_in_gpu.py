'''
演示将多GPU模型加载到cpu
'''
import os
import torch
import torch.nn as nn


class FooNet(nn.Module): # 定义一个全连接网络
    def __init__(self, neural_num, layers=3):
        super(FooNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):

        print("\nbatch size in forward: {}".format(x.size()[0]))

        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        return x


# =================================== 加载至cpu
flag = 0
# flag = 1
if flag:
    gpu_list = [0]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str) # 设置可见的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = FooNet(neural_num=3, layers=3)
    net.to(device)

    # save
    net_state_dict = net.state_dict()
    path_state_dict = "./model_in_gpu_0.pkl"
    torch.save(net_state_dict, path_state_dict)

    # load
    # state_dict_load = torch.load(path_state_dict)
    state_dict_load = torch.load(path_state_dict, map_location="cpu")
    print("state_dict_load:\n{}".format(state_dict_load))


# =================================== 多gpu 保存
flag = 0
# flag = 1
if flag:

    if torch.cuda.device_count() < 2:
        print("gpu数量不足，请到多gpu环境下运行")
        import sys
        sys.exit(0)

    gpu_list = [0, 1, 2, 3]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = FooNet(neural_num=3, layers=3)
    net = nn.DataParallel(net) # 关键代码，设置网络为多GPU并行模式
    net.to(device)

    # save
    net_state_dict = net.state_dict()
    path_state_dict = "./model_in_multi_gpu.pkl"
    torch.save(net_state_dict, path_state_dict)

# =================================== 多gpu 加载
# flag = 0
flag = 1
if flag:

    net = FooNet(neural_num=3, layers=3) # 模型在cpu上

    path_state_dict = "./model_in_multi_gpu.pkl"
    # 加载多GPU版本模型的参数到CPU中
    state_dict_load = torch.load(path_state_dict, map_location="cpu") 
    print("state_dict_load:\n{}".format(state_dict_load))

    # net.load_state_dict(state_dict_load)

    # 移除 'module.'
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict_load.items():
        # 多GPU版本中，模块名多了'module.'，要移除才能在cpu上跑
        namekey = k[7:] if k.startswith('module.') else k 
        new_state_dict[namekey] = v
    print("new_state_dict:\n{}".format(new_state_dict))

    net.load_state_dict(new_state_dict)




















