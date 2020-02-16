# -*- coding: utf-8 -*-
"""
这个文件演示三种交叉熵损失、负对数损失的用法和参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# fake data
inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)

# ----------------------------------- 1 演示交叉熵损失的reduction参数
flag = 0
# flag = 1
if flag:
    # def loss function
    loss_f_none = nn.CrossEntropyLoss(weight=None, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=None, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')

    # forward
    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("Cross Entropy Loss:\n ", loss_none, loss_sum, loss_mean)

# --------------------------------- 手算交叉熵损失
flag = 0
# flag = 1
if flag:

    idx = 0

    input_1 = inputs.detach().numpy()[idx]      # [1, 2]
    target_1 = target.numpy()[idx]              # [0]

    # 第一项
    x_class = input_1[target_1]

    # 第二项
    sigma_exp_x = np.sum(list(map(np.exp, input_1)))
    log_sigma_exp_x = np.log(sigma_exp_x)

    # 输出loss
    loss_1 = -x_class + log_sigma_exp_x

    print("第一个样本loss为: ", loss_1)


# ----------------------------------- 演示weight参数的使用
flag = 0
# flag = 1
if flag:
    # def loss function
    weights = torch.tensor([1, 2], dtype=torch.float)
    # weights = torch.tensor([0.7, 0.3], dtype=torch.float)

    loss_f_none_w = nn.CrossEntropyLoss(weight=weights, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("\nweights: ", weights)
    print(loss_none_w, loss_sum, loss_mean)


# --------------------------------- 手算验证weight参数的使用方法
flag = 0
# flag = 1
if flag:
    weights = torch.tensor([1, 2], dtype=torch.float)
    weights_all = np.sum(list(map(lambda x: weights.numpy()[x], target.numpy())))  # [0, 1, 1]  # [1 2 2]

    mean = 0
    loss_sep = loss_none.detach().numpy()
    for i in range(target.shape[0]):

        x_class = target.numpy()[i]
        tmp = loss_sep[i] * (weights.numpy()[x_class] / weights_all)
        mean += tmp

    print(mean)


# ----------------------------------- 2 NLLLoss 负对数损失函数
flag = 0
# flag = 1
if flag:

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.NLLLoss(weight=weights, reduction='none')
    loss_f_sum = nn.NLLLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.NLLLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("\nweights: ", weights)
    print("NLL Loss", loss_none_w, loss_sum, loss_mean)


# ----------------------------------- 3 BCELoss 用于二分类的交叉熵损失函数
flag = 0
# flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # itarget
    inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCELoss(weight=weights, reduction='none')
    loss_f_sum = nn.BCELoss(weight=weights, reduction='sum')
    loss_f_mean = nn.BCELoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\nweights: ", weights)
    print("BCE Loss", loss_none_w, loss_sum, loss_mean)


# --------------------------------- 手算BCE Loss
flag = 0
# flag = 1
if flag:

    idx = 0

    x_i = inputs.detach().numpy()[idx, idx]
    y_i = target.numpy()[idx, idx]              #

    # loss
    # l_i = -[ y_i * np.log(x_i) + (1-y_i) * np.log(1-y_i) ]      # np.log(0) = nan
    l_i = -y_i * np.log(x_i) if y_i else -(1-y_i) * np.log(1-x_i)

    # 输出loss
    print("BCE inputs: ", inputs)
    print("第一个loss为: ", l_i)


# ----------------------------------- 4 BCEWithLogitsLoss 即结合了sigmoid的二分类交叉熵损失函数
# flag = 0
flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1, 1], dtype=torch.float)

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\nweights: ", weights)
    print(loss_none_w, loss_sum, loss_mean)


# --------------------------------- 演示pos_weight参数，即正样本权值，用于平衡正负样本

# flag = 0
flag = 1
if flag:
    inputs = torch.tensor([[1, 2], [2, 2], [3, 4], [4, 5]], dtype=torch.float)
    target = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)

    target_bce = target

    # itarget
    # inputs = torch.sigmoid(inputs)

    weights = torch.tensor([1], dtype=torch.float)
    pos_w = torch.tensor([3], dtype=torch.float)        # 3

    loss_f_none_w = nn.BCEWithLogitsLoss(weight=weights, reduction='none', pos_weight=pos_w)
    loss_f_sum = nn.BCEWithLogitsLoss(weight=weights, reduction='sum', pos_weight=pos_w)
    loss_f_mean = nn.BCEWithLogitsLoss(weight=weights, reduction='mean', pos_weight=pos_w)

    # forward
    loss_none_w = loss_f_none_w(inputs, target_bce)
    loss_sum = loss_f_sum(inputs, target_bce)
    loss_mean = loss_f_mean(inputs, target_bce)

    # view
    print("\npos_weights: ", pos_w)
    print(loss_none_w, loss_sum, loss_mean)


