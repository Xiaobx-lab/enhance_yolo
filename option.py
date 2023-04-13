import argparse
import random
import torch
import os
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser(description='ConMixer')
parser.add_argument('--test_see', default=True,type=bool, help='batch size')
parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--epoch', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--init_lr', default=2e-4, type=float, help='a low initial learning rata for adamw optimizer')
parser.add_argument('--wd', default=0.5, type=float, help='a high weight decay setting for adamw optimizer')
parser.add_argument('--model_dir', default='weight/', type=str, help='the path to saving the checkpoints')
parser.add_argument('--save_best', default=True, type=bool, help='saveing the checkpoint has the best acc')
parser.add_argument('--crop_size', type=int, default=256, help='Takes effect when using --crop ')
opt = parser.parse_args()
opt.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

batch_size = opt.batch_size
crop_size = opt.crop_size

if __name__ == "__main__":
    pass
