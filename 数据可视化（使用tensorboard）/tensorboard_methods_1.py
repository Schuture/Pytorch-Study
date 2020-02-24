"""
tensorboard方法使用1 scalars and histogram
add_scalar / add_scalars方法创建曲线/单张图多曲线
add_histogram方法添加三维的直方图
"""
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子

# ----------------------------------- 0 SummaryWriter -----------------------------------
flag = 0
# flag = 1
if flag:

    log_dir = "./train_log/test_log_dir"
    # writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix="12345678")
    writer = SummaryWriter(comment='_scalars', filename_suffix="12345678")

    for x in range(100):
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

    writer.close()


# ----------------------------------- 1 scalar and scalars -----------------------------------
flag = 0
# flag = 1
if flag:

    max_epoch = 100

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(max_epoch):

        writer.add_scalar('y=2x', x * 2, x)
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

        writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                                 "xcosx": x * np.cos(x)}, x)

    writer.close()

# ----------------------------------- 2 histogram -----------------------------------
# flag = 0
flag = 1
if flag:

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(2):

        np.random.seed(x)

        data_unif = np.arange(100)
        data_normal = np.random.normal(size=1000)

        writer.add_histogram('distribution uniform', data_unif, x)
        writer.add_histogram('distribution normal', data_normal, x)

        plt.subplot(121).hist(data_unif, label="uniform")
        plt.subplot(122).hist(data_normal, label="normal")
        plt.legend()
        plt.show()

    writer.close()
