import os
import os.path as osp
from argparse import ArgumentParser
from time import strftime

import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import RandAffine, RandFlip, Compose, NormalizeIntensity, RandBiasField, ToTensor, Resize
from PIL import Image, ImageDraw

from models.baseline import *
from models.vit import *
from models.resnet3d import ResNet3d as ResNet
from dataset.brain_p1_flair import *
from utils import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--bs', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    # parser.add_argument('--model', default='resnet')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_dir', default='logs/default')
    # parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--ckpt', default='/data/ssd_2/liuyy/BrainAgeLujm/Brain_Baseline/brain-baseline/logs/resnet-20240304155316/20240305154007-e127-6.8778.ckpt', type=str)
    parser.add_argument('--tag', default=None, type=str)

    args = parser.parse_args()
    return args


def build_model(args):
    model = ResNet()
    return model


def load_ckpt(args, model, optim=None, lr_scheduler=None):
    if args.ckpt is None:
        return 1, 1000

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optim is not None:
        optim.load_state_dict(ckpt['optim'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    print(f"load ckpt from {args.ckpt}")

    return ckpt['epoch'] + 1, ckpt['mae']


@torch.no_grad()
def test(args, model):
    load_ckpt(args, model)
    model.eval()

    test_set = BrainSet(mode='test')
    test_loader = DataLoader(test_set, args.bs, num_workers=4, shuffle=False)

    mae_m = AverageMeter()

    for step, sample in enumerate(test_loader):
        data, label = sample['data'].cuda(), sample['label'].cuda()
        # output = model(data)
        output = model(data).reshape(label.shape)

        mae = torch.abs(output - label)
        mae_m.add(mae)

    print(f'Test result for INTER > MAE:{mae_m.value:.4f}, {len(test_set)} cases')

if __name__ == '__main__':
    args = parse_args()

        # args.log_dir = args.log_dir + '-' + strftime("%Y%m%d%H%M%S")
    model = build_model(args).cuda()
    archives = ['INTER']

    for archive in archives:
        test(args, model)
