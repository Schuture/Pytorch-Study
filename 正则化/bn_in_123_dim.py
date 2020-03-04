# -*- coding: utf-8 -*-
"""
bn权值初始化，123维bn分别用在feature map维度为123时
"""
import torch
import torch.nn as nn
from tools.common_tools import set_seed



set_seed(1)  # 设置随机种子

# ======================================== nn.BatchNorm1d
flag = 1
#flag = 0
if flag:

    batch_size = 3
    num_features = 5
    momentum = 0.1 # 修改这里的momentum，可以观察到bn层是如何将计算结果传递给下一次迭代的

    features_shape = (1)

    feature_map = torch.ones(features_shape)                                                    # 1D，1x1特征图
    feature_maps = torch.stack([feature_map*(i+1) for i in range(num_features)], dim=0)         # 2D，加上了通道维度
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)             # 3D，加上了batch维度
    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    bn = nn.BatchNorm1d(num_features=num_features, momentum=momentum) # 一维的BN相当于全连接层在相同位置的元素上归一化

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)

        print("\niteration:{}, running mean: {} ".format(i, bn.running_mean))
        print("iteration:{}, running var:{} ".format(i, bn.running_var))

        mean_t, var_t = 2, 0

        running_mean = (1 - momentum) * running_mean + momentum * mean_t
        running_var = (1 - momentum) * running_var + momentum * var_t

        print("iteration:{}, 第二个特征的running mean: {} ".format(i, running_mean))
        print("iteration:{}, 第二个特征的running var:{}".format(i, running_var))


# ======================================== nn.BatchNorm2d
#flag = 1
flag = 0
if flag:

    batch_size = 3
    num_features = 6
    momentum = 0.3
    
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)                                                    # 2D
    feature_maps = torch.stack([feature_map*(i+1) for i in range(num_features)], dim=0)         # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)             # 4D

    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    bn = nn.BatchNorm2d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)

        print("\niter:{}, running_mean.shape: {}".format(i, bn.running_mean.shape))
        print("iter:{}, running_var.shape: {}".format(i, bn.running_var.shape))

        print("iter:{}, weight.shape: {}".format(i, bn.weight.shape))
        print("iter:{}, bias.shape: {}".format(i, bn.bias.shape))

# ======================================== nn.BatchNorm3d
#flag = 1
flag = 0
if flag:

    batch_size = 3
    num_features = 4
    momentum = 0.3

    features_shape = (2, 2, 3)

    feature = torch.ones(features_shape)                                                # 3D
    feature_map = torch.stack([feature * (i + 1) for i in range(num_features)], dim=0)  # 4D
    feature_maps = torch.stack([feature_map for i in range(batch_size)], dim=0)         # 5D

    print("input data:\n{} shape is {}".format(feature_maps, feature_maps.shape))

    bn = nn.BatchNorm3d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps)

        print("\niter:{}, running_mean.shape: {}".format(i, bn.running_mean.shape))
        print("iter:{}, running_var.shape: {}".format(i, bn.running_var.shape))

        print("iter:{}, weight.shape: {}".format(i, bn.weight.shape))
        print("iter:{}, bias.shape: {}".format(i, bn.bias.shape))






