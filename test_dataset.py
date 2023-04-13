import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.utils import save_image

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import cvtColor, preprocess_input, get_classes, get_anchors

if __name__ == "__main__":

    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'
    input_shape = [416, 416]
    loss_value_all = 0
    Cuda = False
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # 读取数据集
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    # num_train   = len(train_lines)
    # num_val     = len(val_lines)

    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    # 获取classes和anchors
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=True)

    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    model = YoloBody(3,3,anchors_mask, num_classes)
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    # print(len(gen))

    for iteration, batch in enumerate(gen_val):
        images, images_clear,targets = batch[0], batch[1], batch[2]  # 图像 标签（锚框坐标，类别）
        # outputs = model(images)               # 输出三张特征图
        # for l in range(len(outputs)):
        #     save_image(batch[0],'低光图像.jpg')
        #     save_image(batch[1], '清晰图像.jpg')
        #     loss_item = yolo_loss(l, outputs[l], targets)
        #     loss_value_all += loss_item
        # loss_value = loss_value_all

        save_image(images, 'input_test.jpg')
        save_image(images_clear,'clear_test.jpg')
        break
        # print(iteration)
