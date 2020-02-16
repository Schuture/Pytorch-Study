1_split_dataset.py用于将数据集分割为训练、验证、测试三个部分，使用时修改路径，路径可以为人民币数据集或者猫狗数据集

2_train_lenet.py用于训练一个LeNet神经网络来进行图片的二分类，使用时设置Dataset类为RMBDataset或者CatDogDataset，并修改数据集路径

my_transforms.py定义了一个增加椒盐噪声的transforms类，可以添加到torchvision.transforms.Compose()中使用

RMB_data_augmentation.py使用数据增强的方法来训练人民币分类网络，使得在第四套人民币上训练的模型也能够分类第五套人民币

transforms_methods_1/2.py是两个展示数据增强方法的demo，可以取消某些注释来观察transforms方法的结果

猫狗数据集和人民币数据集：
猫狗链接：https://pan.baidu.com/s/1-0xHuViQz7lir6o1KxgJxg 密码：6837；
RMB链接：https://pan.baidu.com/s/1tzqsnawtFOOdAhsQ_1SEEw 提取码：gu8g 