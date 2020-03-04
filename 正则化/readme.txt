***L2_regularization.py使用两个分别有/没有L2正则化的网络进行训练，对比它们的参数，发现有正则化会使得参数更集中在0附近

***dropout_layer.py展示dropout层如何添加到模型中，训练与测试有什么异同

***dropout_regularization.py 使用dropout在自己生成的数据上做实验，发现有dropout的模型拟合的曲线更加平滑，不会过拟合

***bn_and_initialize.py 对比了使用初始化方法与使用bn层，数据的稳定性有什么差异，结论是bn层更稳定

***bn_application.py 在猫狗分类数据集上应用了带bn层的LeNet

***bn_in_123_dim.py 展示了bn权值初始化，123维bn分别用在feature map维度为123时

***normalization_layers.py 展示了pytorch中常见的 normalization layers，LN, IN, GN