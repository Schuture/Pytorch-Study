import os
import random
import shutil


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)

    dataset_dir = os.path.join("data", "CatDog_data")
    split_dir = os.path.join("data", "CatDog_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):
        for sub_dir in dirs: # 从两个子文件夹中分别读取不同类别的数据

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct) # 训练集大小
            valid_point = int(img_count * (train_pct + valid_pct)) # 训练集+验证集大小

            for i in range(img_count):
                if i < train_point: # 这些分在训练集中
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point: # 这些分在验证集中
                    out_dir = os.path.join(valid_dir, sub_dir)
                else: # 这些分在测试集中
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path) # 把图片从原路径复制到目标路径

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
