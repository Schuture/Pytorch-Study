"""
数据、模型迁移至cuda的方法，演示在GPU中进行模型前向传播
"""
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========================== tensor to cuda
# flag = 0
flag = 1
if flag:
    x_cpu = torch.ones((3, 3))
    print("x_cpu:\ndevice: {} is_cuda: {} id: {}".format(x_cpu.device, x_cpu.is_cuda, id(x_cpu)))

    x_gpu = x_cpu.to(device) # tensor迁移到cuda需要赋值，不是原地操作
    print("x_gpu:\ndevice: {} is_cuda: {} id: {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu)))

# 弃用
# x_gpu = x_cpu.cuda()

# ========================== module to cuda
# flag = 0
flag = 1
if flag:
    net = nn.Sequential(nn.Linear(3, 3))

    print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

    net.to(device) # 模型迁移到cuda不需要赋值，是原地操作
    print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))


# ========================== forward in cuda
#flag = 0
flag = 1
if flag:
    output = net(x_gpu)
    print("output is_cuda: {}".format(output.is_cuda))

    # output = net(x_cpu)

# ========================== 查看当前gpu 序号，尝试修改可见gpu，以及主gpu
flag = 0
#flag = 1
if flag:
    current_device = torch.cuda.current_device() # 当前设备
    print("current_device: ", current_device)

    torch.cuda.set_device(0) # 将设备设置为0号逻辑GPU
    current_device = torch.cuda.current_device()
    print("current_device: ", current_device)


    # 设备容量
    cap = torch.cuda.get_device_capability(device=None)
    print(cap)
    
    # 设备名
    name = torch.cuda.get_device_name()
    print(name)

    is_available = torch.cuda.is_available()
    print(is_available)

    # ===================== seed，为cuda设备设置随机种子
    seed = 2
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    current_seed = torch.cuda.initial_seed() # 获取当前随机种子
    print(current_seed)

    s = torch.cuda.seed() # 设置随机数种子
    s_all = torch.cuda.seed_all()
    print(s, s_all)




