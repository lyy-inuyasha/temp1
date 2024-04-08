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
from dataset.brain import *
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
    # parser.add_argument('--ckpt', default='/data/ssd_2/liuyy/Brain_Baseline/brain-baseline/logs/sfcn-gm_lr0.0001_bs8/20230516190207/20230517092430-e110-2.9050.ckpt', type=str)
    parser.add_argument('--ckpt', default='/data/ssd_2/liuyy/Brain_Baseline/brain-baseline/logs/SFCN/20240224101318/20240226180359-e174-4.2297.ckpt', type=str)
    parser.add_argument('--tag', default=None, type=str)

    args = parser.parse_args()
    return args


def build_model(args):
    model = SFCN()
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
def test(args, model, archive):
    load_ckpt(args, model)
    model.eval()

    test_transform = Compose([
        ToTensor(),
        NormalizeIntensity(channel_wise=True)
    ])
    test_set = DisBrainTestSet(npz=archive, transform=test_transform)
    # test_set = DisBrainSet(mode='valid', transform=test_transform)
    test_loader = DataLoader(test_set, args.bs, num_workers=8, shuffle=False)

    mae_m = AverageMeter()

    for step, sample in enumerate(test_loader):
        data, label = sample['data'].cuda(), sample['label'].cuda()
        output = model(data)
        output = output.reshape(label.shape)

        prob = F.softmax(output, dim=1)
        pred = test_loader.dataset.discreter.dis2contin(prob).cpu()
        mae = torch.abs(pred - sample['gt'])
        mae_m.add(mae)

    # args.logger.info(f'[TEST] ckpt:{osp.join(args.log_dir, args.ckpt)} | MAE:{mae_m.value:.4f}')
    print(f'Test result for {archive} > MAE:{mae_m.value:.4f}, {len(test_set)} cases')

if __name__ == '__main__':
    args = parse_args()

        # args.log_dir = args.log_dir + '-' + strftime("%Y%m%d%H%M%S")
    model = build_model(args).cuda()

    # archives = ['NIFD', 'INTER']
    archives = ['INTER']

    for archive in archives:
        test(args, model, archive)
