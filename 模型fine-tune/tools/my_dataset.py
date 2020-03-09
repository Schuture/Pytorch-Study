import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
label_name = {"ants": 0, "bees": 1}

class AntsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        蚂蚁蜜蜂分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"ants": 0, "bees": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index): # 根据index返回数据
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self): # 查看样本的数量
        return len(self.data_info)

    @staticmethod # 静态方法可以使得函数直接不通过实例化就调用，例如A.static_foo(1)
    def get_img_info(data_dir): # 自己定义的用来读取数据的函数
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs: # 遍历类别目录
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                for i in range(len(img_names)): # 遍历图片
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = label_name[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info
