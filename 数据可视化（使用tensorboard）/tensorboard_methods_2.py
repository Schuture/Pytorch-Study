"""
tensorboard方法使用2
add_image显示图像，make_grid+add_image显示网格排布的多张图像
add_graph显示网络计算图
"""
import os
import torch
import time
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tools.my_dataset import RMBDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tools.common_tools import set_seed
from model.lenet import LeNet


set_seed(1)  # 设置随机种子


# ----------------------------------- 3 image -----------------------------------
flag = 0
# flag = 1
if flag:

    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # img 1     正态随机图像
    fake_img = torch.randn(3, 512, 512)
    writer.add_image("fake_img", fake_img, 1)
    time.sleep(1)

    # img 2     全1图像，被识别为0-1像素范围，白色
    fake_img = torch.ones(3, 512, 512)
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 2)

    # img 3     全1.1图像，被识别为0-255像素范围，黑色
    fake_img = torch.ones(3, 512, 512) * 1.1
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 3)

    # img 4     HW，灰度图
    fake_img = torch.rand(512, 512)
    writer.add_image("fake_img", fake_img, 4, dataformats="HW")

    # img 5     HWC，彩色图
    fake_img = torch.rand(512, 512, 3)
    writer.add_image("fake_img", fake_img, 5, dataformats="HWC")

    writer.close()


# ----------------------------------- 4 make_grid -----------------------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    split_dir = os.path.join("data", "RMB_split")
    train_dir = os.path.join(split_dir, "train")
    # train_dir = "path to your training data"

    # 定义一个数据提取器
    transform_compose = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor()])
    train_data = RMBDataset(data_dir=train_dir, transform=transform_compose)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    data_batch, label_batch = next(iter(train_loader)) # 提取一个batch的数据

    # 一个4*4的grid。根据输入图像的像素值范围，决定是否使用normalize
    img_grid = vutils.make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
    # img_grid = vutils.make_grid(data_batch, nrow=4, normalize=False, scale_each=False)
    writer.add_image("input img", img_grid, 0)

    writer.close()


# ----------------------------------- 5 add_graph -----------------------------------

# flag = 0
flag = 1
if flag:

    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # 模型
    fake_img = torch.randn(1, 3, 32, 32)

    lenet = LeNet(classes=2)

    # 可视化模型计算图
    writer.add_graph(lenet, fake_img)

    writer.close()

    # 打印模型各层信息
    from torchsummary import summary
    print(summary(lenet, (3, 32, 32), device="cpu"))











