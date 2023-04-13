from collections import OrderedDict
from typing_extensions import Self

import torch
import torch.nn as nn
import numpy as np
from nets.AFS import AFS
from nets.darknet import darknet53
from nets.unet_parts import DoubleConv, Down, Up, OutConv


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self,  n_channels, n_classes, anchors_mask, num_classes, bilinear=False, pretrained = True):
        super(YoloBody, self).__init__()
        #--------------------------------Unet-----------------------------------------------
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.sigmoid = torch.nn.Sigmoid()
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        out_filters = self.backbone.layers_out_filters
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x, control_AFS8 = None, control_AFS_16 = None):

        #低光增强分支
        x1 = self.inc(x)         # x1-> 1,32,416,416
        x2 = self.down1(x1)      # x2-> 1,64,208,208
        x3 = self.down2(x2)      # x3-> 1,128,104,104
        x4 = self.down3(x3)      # x4-> 1,256,52,52
        x5 = self.down4(x4)      # x5-> 1,512,26,26
        
        #检测
        x2_, x1_, x0_ = self.backbone(x) # 8 16 32

        # AFS交互
        if control_AFS_16 is not None:
            high_map16, low_map16 = control_AFS_16(x1_, x5)
            x1_ = x1_ + low_map16
            x5 = x5 + high_map16

        # 增强
        up1 = self.up1(x5, x4)   # 1,256,52,52

        # AFS交互
        if control_AFS8 is not None:
            high_map8, low_map8 = control_AFS8(x2_, up1)
            x2_ = x2_ + low_map8
            up1 = up1 + high_map8
       
        # 增强
        up2 = self.up2(up1, x3)  # 1,128,104,104
        up3 = self.up3(up2, x2)  # 1,64,208,208
        up4 = self.up4(up3, x1)  # 1,32,416,416
        logits = self.outc(up4)
        enhance_out = self.sigmoid(logits)

        # 检测
        out0_branch = self.last_layer0[:5](x0_)
        out0        = self.last_layer0[5:](out0_branch) # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)         # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = torch.cat([x1_in, x1_], 1)               # 26,26,256 + 26,26,512 -> 26,26,768
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)  # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in) # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = torch.cat([x2_in, x2_], 1)       # 52,52,128 + 52,52,256 -> 52,52,384
        out2 = self.last_layer2(x2_in)           # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128

        return enhance_out, out0, out1, out2


if __name__ == "__main__":
    X = torch.Tensor(1,3,416,416)
    net = YoloBody(3,3,anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=20, bilinear=False)

    outputs = net(X)
