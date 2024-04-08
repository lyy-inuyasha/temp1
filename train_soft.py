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
# from dataset.brain import *
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
    parser.add_argument('--log_dir', default='logs/SFCN')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--tag', default=None, type=str)

    args = parser.parse_args()
    return args


def build_model(args):
    model = SFCN()
    return model


def load_ckpt(args, model, optim=None, lr_scheduler=None):
    if args.ckpt is None:
        return 1, 1000

    ckpt = torch.load(osp.join(args.log_dir, args.ckpt), map_location='cpu')
    model.module.load_state_dict(ckpt['model'])
    if optim is not None:
        optim.load_state_dict(ckpt['optim'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    if args.local_rank == 0:
        args.logger.info(f"load ckpt from {osp.join(args.log_dir, args.ckpt)}")

    return ckpt['epoch'] + 1, ckpt['mae']


def save_ckpt(args, epoch, mae, model, optim, lr_scheduler=None):
    ckpt_dict = {
        'model':model.module.state_dict(),
        'optim':optim.state_dict(),
        'mae':mae,
        'epoch':epoch,
    }
    if lr_scheduler is not None:
        ckpt_dict['lr_scheduler'] = lr_scheduler.state_dict()

    if args.ckpt is not None:
        old_ckpt_path = osp.join(args.log_dir, args.ckpt)
        os.remove(old_ckpt_path)

    ckpt_file = f'{strftime("%Y%m%d%H%M%S")}-e{epoch}-{mae:.4f}.ckpt'
    args.ckpt = ckpt_file
    args.logger.info(f"save model to {osp.join(args.log_dir, args.ckpt)}")
    torch.save(ckpt_dict, osp.join(args.log_dir, args.ckpt))


def train_epoch(args, epoch, model, loss_fn, optim, dataloader, lr_scheduler=None):
    model.train()
    dataloader.sampler.set_epoch(epoch)

    mae_m, loss_m = AverageMeter(), AverageMeter()
    calc_m, read_m = AverageMeter(), AverageMeter()
    timer = Timer()
    log_step = len(dataloader) // 11
    if args.local_rank == 0:
        args.writer.add_scalar('lr', optim.param_groups[0]['lr'], epoch)

    mae_list, pred_list = [], []

    for step, sample in enumerate(dataloader):
        data, label = sample['data'].cuda(), sample['label'].cuda()
        read_m.add(timer.tiktok())

        optim.zero_grad()
        # (output, deep_output), attn = model(data)
        output = model(data)
        output = output.reshape(label.shape)
        # loss = loss_fn(output, label) + loss_fn(deep_output, label)
        loss = loss_fn(output, label)
        loss.backward()

        optim.step()

        calc_m.add(timer.tiktok())
        prob = F.softmax(output, dim=1)
        pred = dataloader.dataset.discreter.dis2contin(prob).cpu()
        mae = torch.abs(pred - sample['gt'])
        pred_list.append(pred)
        mae_list.append(mae)
        mae_m.add(mae)
        loss_m.add(loss.item(), data.shape[0])

        if args.local_rank == 0:
            global_step = (epoch - 1) * len(dataloader) + step
            args.writer.add_scalar('Loss/train', loss.item(), global_step)
            args.writer.add_scalar('MAE/train', mae.mean(), global_step)
            # args.writer.add_histogram('Image_stat/train', data[0].flatten(), global_step)
            if global_step % log_step == 0:
                args.logger.info(f'[TRAIN] Epoch {epoch}@{step/len(dataloader)*100:.1f}% | loss:{loss_m.value:.4f}, MAE:{mae_m.value:.4f}, read_time:{read_m.value:.3f}s, calc_time:{calc_m.value:.3f}s')
                slice_idx = torch.arange(0, data.shape[-1], data.shape[-1]//6)
                slices = data[0, 0, ..., slice_idx].cpu()
                slices = (slices - slices.min()) / (slices.max() - slices.min())
                label_slice = Image.new('F', (slices.shape[1], slices.shape[0]))
                d = ImageDraw.Draw(label_slice)
                d.text((10, 10), str(sample['gt'][0]))
                label_slice = torch.tensor(np.array(label_slice))
                slices[..., -1] += label_slice
                slices = torch.stack([slices]*3)
                step = global_step // log_step
                args.writer.add_images('Images/train', slices, step, dataformats='CHWN')

                # attn_scores = attn[0, -1]
                # attn_scores = torch.stack([attn_scores]*3)
                # args.writer.add_images('Attn/train', attn_scores, step, dataformats='CNHW')

                # write model grad
                for name, param in model.named_parameters():
                    args.writer.add_scalar(f'WeightNorm/{name}', torch.linalg.norm(param), step)
                    args.writer.add_scalar(f'GradNorm/{name}', torch.linalg.norm(param.grad), step)


        timer.tiktok()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if args.local_rank == 0:
        args.writer.add_histogram('MAE_stat/train', torch.cat(mae_list).flatten(), epoch)
        args.writer.add_histogram('pred_stat/train', torch.cat(pred_list).flatten(), epoch)
        args.writer.add_histogram('Image_stat/train', data[0].flatten(), epoch)
        args.logger.info(f'[TRAIN] Epoch {epoch}@100% | loss:{loss_m.value:.4f}, MAE:{mae_m.value:.4f}, read_time:{read_m.value:.3f}s, calc_time:{calc_m.value:.3f}s')


@torch.no_grad()
def val_epoch(args, epoch, model, loss_fn, dataloader):
    if args.local_rank != 0:
        return None
    model.eval()
    mae_m, loss_m = AverageMeter(), AverageMeter()
    mae_list = []
    pred_list = []

    for step, sample in enumerate(dataloader):
        data, label = sample['data'].cuda(), sample['label'].cuda()

        # (output, deep_out), attn = model(data)
        output = model(data)
        output = output.reshape(label.shape)
        # loss = loss_fn(output, label) + loss_fn(deep_out, label)
        loss = loss_fn(output, label)

        prob = F.softmax(output, dim=1)
        pred = dataloader.dataset.discreter.dis2contin(prob).cpu()
        mae = torch.abs(pred - sample['gt'])
        mae_list.append(mae)
        pred_list.append(pred)
        mae_m.add(mae)
        loss_m.add(loss.item(), data.shape[0])

    args.writer.add_scalar('Loss/valid', loss_m.value, epoch)
    args.writer.add_scalar('MAE/valid', mae_m.value, epoch)
    args.writer.add_histogram('MAE_stat/valid', torch.cat(mae_list).flatten(), epoch)
    args.writer.add_histogram('pred_stat/valid', torch.cat(pred_list).flatten(), epoch)
    args.logger.info(f'[VALID] Epoch {epoch} | loss:{loss_m.value:.4f}, MAE:{mae_m.value:.4f}')

    return mae_m.value


@torch.no_grad()
def test(args, model):
    if args.local_rank != 0:
        return

    load_ckpt(args, model)
    model.eval()

    test_transform = Compose([
        ToTensor(),
        NormalizeIntensity(channel_wise=True)
    ])
    test_set = DisBrainSet(mode='test', transform=test_transform)
    test_loader = DataLoader(test_set, args.bs, num_workers=8, shuffle=False)

    mae_m = AverageMeter()

    for step, sample in enumerate(test_loader):
        data, label = sample['data'].cuda(), sample['label'].cuda()
        # (output, deep_out), _ = model(data)
        output = model(data)
        output = output.reshape(label.shape)

        prob = F.softmax(output, dim=1)
        pred = test_loader.dataset.discreter.dis2contin(prob).cpu()
        mae = torch.abs(pred - sample['gt'])
        mae_m.add(mae)

    args.logger.info(f'[TEST] ckpt:{osp.join(args.log_dir, args.ckpt)} | MAE:{mae_m.value:.4f}')


def train(args):
    train_transform = Compose([
        ToTensor(),
        RandFlip(prob=0.5, spatial_axis=0),
        RandAffine(prob=1, translate_range=(2, 2, 0)),
        RandBiasField(coeff_range=(.1, .2), prob=.5),
        NormalizeIntensity(channel_wise=True)
    ])
    dev_transform = Compose([
        ToTensor(),
        NormalizeIntensity(channel_wise=True)
    ])
    train_set = DisBrainSet(mode='train', transform=train_transform)
    dev_set = DisBrainSet(mode='val', transform=dev_transform)

    model = build_model(args)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, args.bs, pin_memory=False, num_workers=8, sampler=train_sampler, drop_last=True)
    dev_loader = DataLoader(dev_set, args.bs, num_workers=8, pin_memory=False, shuffle=False)

    def loss_fn(output, target):
        if target.shape[0] != output.shape[0]:
            w = output.shape[0] // target.shape[0]
            target = target.unsqueeze(1).repeat(1, w, 1).reshape(output.shape[0], -1)
        output = F.log_softmax(output, dim=1)
        loss = F.kl_div(output, target, reduction='batchmean')
        return loss

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.995)
    lr_scheduler = None

    epoch_, mae = load_ckpt(args, model, optim, lr_scheduler)

    for epoch in range(epoch_, args.epochs + 1):
        train_epoch(args, epoch, model, loss_fn, optim, train_loader, lr_scheduler)
        val_mae = val_epoch(args, epoch, model, loss_fn, dev_loader)
        if val_mae is not None and val_mae < mae:
            save_ckpt(args, epoch, val_mae, model, optim, lr_scheduler)
            mae = val_mae
    test(args, model)


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed+args.local_rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5679'
    dist.init_process_group(backend='gloo', init_method='env://', rank=0,
                            world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)

    # dist.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)
    dist.barrier()

    if args.local_rank == 0:
        # args.log_dir = args.log_dir + '-' + strftime("%Y%m%d%H%M%S")
        if args.ckpt is None:
            args.log_dir = osp.join(args.log_dir, strftime("%Y%m%d%H%M%S"))
            os.makedirs(args.log_dir, exist_ok=True)
            # os.rename('a.sh', osp.join(args.log_dir, 'train.sh'))
        print(args.log_dir)

        args.writer = SummaryWriter(args.log_dir)
        args.logger = build_logger(args.log_dir, 'train.log')

        args.logger.info(args)

    train(args)
