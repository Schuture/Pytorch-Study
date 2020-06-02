# -*- coding: utf-8 -*-
"""
实现Faster rcnn目标检测的inference
"""

import os
import time
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# coco数据集的类别，91类，其中1类背景，90类物体
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


if __name__ == "__main__":

    path_img = os.path.join(BASE_DIR, "demo_img1.png")
    #path_img = os.path.join(BASE_DIR, "demo_img2.png")

    # config
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 1. load data & model
    input_image = Image.open(path_img).convert("RGB")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 2. preprocess
    img_chw = preprocess(input_image)

    # 3. to device
    if torch.cuda.is_available():
        img_chw = img_chw.to(device)
        model.to(device)

    # 4. forward
    input_list = [img_chw]
    with torch.no_grad():
        tic = time.time()
        print("input img tensor shape:{}".format(input_list[0].shape))
        output_list = model(input_list)
        output_dict = output_list[0] # 返回的也是一个tensor，第一个维度对应不同的数据样本
        print("pass: {:.3f}s".format(time.time() - tic))
        for k, v in output_dict.items():
            print("key:{}, value:{}".format(k, v))

    # 5. visualization
    out_boxes = output_dict["boxes"].cpu() # 二维tensor存储框位置[[x1, y1, x2, y2], ...]
    out_scores = output_dict["scores"].cpu() # 一维tensor存储每一个框的置信度
    out_labels = output_dict["labels"].cpu() # 一维tensor存储每一个框的标签

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(input_image, aspect='equal')

    num_boxes = out_boxes.shape[0]
    max_vis = 40 # 最多取40个边界框
    thres = 0.5 # 阈值大于它的边界框才进行可视化

    for idx in range(0, min(num_boxes, max_vis)):

        score = out_scores[idx].numpy()
        bbox = out_boxes[idx].numpy()
        class_name = COCO_INSTANCE_CATEGORY_NAMES[out_labels[idx]]

        if score < thres: # 如果当前边界框置信度过低，就不标出
            continue

        # 在图中标出方框
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                   edgecolor='red', linewidth=3.5))
        # 在方框上标出类别
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.savefig(path_img.split('.')[0] + '_with_bbox.png')
    plt.show()
    plt.close()

    # appendix，Pascal VOC数据集的类别，共21类，一类背景，20类物体
    classes_pascal_voc = ['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']


