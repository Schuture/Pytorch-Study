***test_tensorboard.py：
用于在安装好tensorboard / tensorboardX后检验是否可用

***tensorboard_methods_1.py ：
展示SummaryWriter类的使用方法，以及最常用的add_scalar / add_scalars / add_histogram方法的使用

***tensorboard_methods_2.py：
展示add_image如何显示图像；
torchvision.utils.make_grid如何使用一个batch的数据建立网格图像；
add_graph如何展示一个构建好的模型的计算图；
此外还有一个十分好用的小工具torchsummary用于展示模型各层的参数量、模型参数总量以及一次推理的内存消耗量

***loss_acc_weights_grad.py：
展示了一个模型训练后，如何将训练过程中的Loss, accuracy, 以及模型参数的权重、梯度变化进行展示

***weight_fmap_visualization.py：
用于展示训练好的模型的卷积核的权重、特征图变化

***hook_methods.py：
展示钩子函数的使用方法

***hook_fmap_vis.py：
展示如何使用钩子方法来可视化特征图