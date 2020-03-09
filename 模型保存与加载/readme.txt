*** model_save.py 演示了如何保存模型 / 保存模型参数

*** model_load.py 演示了如何加载整个网络 / 仅加载网络参数

*** save_checkpoint.py 模拟训练意外停止，使用一个字典来保存断点处的模型参数、优化器参数、epoch这三个必要的用于恢复训练的数据

*** checkpoint_resume.py 接着save_checkpoint.py，使用torch.load()函数载入断点，恢复训练