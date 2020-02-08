import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from tools.common_tools import transform_invert, set_seed

set_seed(1)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================
flag = '3d'

# ================ 2d
if flag == '2d':
    conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)
    
# ================ 3d
if flag == '3d':
    conv_layer = nn.Conv3d(3, 1, (1, 3, 3), padding = (1, 0, 0))
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_tensor.unsqueeze_(dim = 2) # B*C*H*W to B*C*D*H*W 一般三维卷积对5维张量（多了时间维度）计算
    img_conv = conv_layer(img_tensor)

# ================ transposed
if flag == 't':
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)


# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))

if flag == '3d':
    img_raw = transform_invert(img_tensor.squeeze(), img_transform)
    img_conv = transform_invert(img_conv.squeeze(), img_transform)
else:
    img_raw = transform_invert(img_tensor.squeeze(), img_transform)
    img_conv = transform_invert(img_conv[0,0:1,...], img_transform) # 0:1可以保留这个维度

plt.figure(figsize = (16, 16))
plt.subplot(121).imshow(img_raw)
plt.subplot(122).imshow(img_conv, cmap='gray')

plt.show()



