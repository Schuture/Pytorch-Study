import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}
catdog_label = {"cat": 0, "dog": 1}

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"1": 0, "100": 1}
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
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


class CatDogDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        """
        猫狗分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"cat": 0, "dog": 1}
        self.data_info = self.get_img_info(data_dir) # 从路径读取数据信息（图片路径、类别）
        self.transform = transform # 数据转换
        
    def __getitem__(self, index): # 根据index返回数据
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self): # 查看样本的数量
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir): # 自己定义的用来读取数据的函数
        data_info = list()
        for root, dirs, _ in os.walk(data_dir): # train, test, valid
            for sub_dir in dirs: # 两个文件夹，分别为猫、狗
                img_names = os.listdir(os.path.join(root, sub_dir)) # 所有图片的路径
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names)) # 筛选出jpg图片

                for i in range(len(img_names)): # 遍历jpg图片
                    img_name = img_names[i] # 第i张图片
                    path_img = os.path.join(root, sub_dir, img_name) # 第i张图片的完整路径
                    label = catdog_label[sub_dir] # 这张图片的类别
                    data_info.append((path_img, int(label))) # list中添加图片路径和类别

        return data_info
        
        
        
        