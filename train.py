import os
import os.path as osp
from argparse import ArgumentParser
from time import strftime

import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# from models.baseline import ResNet
# from models.resnet3d import ResNet3d as resnet152
from monai.networks.nets.resnet import resnet152
from dataset.brain_p1_flair_region import *
from utils import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--bs', default=200, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--wd', default=0.05, type=float)
    # parser.add_argument('--model', default='resnet')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_dir', default='logs/resnet_region')
    parser.add_argument('--local_rank', default=1, type=int)
    parser.add_argument('--ckpt', default=None, type=str)

    args = parser.parse_args()
    return args


def build_model(args):
    return resnet152(pretrained=False, spatial_dims=3, n_input_channels=2, num_classes=1)


def load_ckpt(args, model, optim=None, lr_scheduler=None):
    if args.ckpt is None:
        return 1, 1000

    ckpt = torch.load(osp.join(args.log_dir, args.ckpt), map_location='cpu')
    model.module.load_state_dict(ckpt['model'])
    if optim is not None:
        optim.load_state_dict(ckpt['optim'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    if args.local_rank == 0 or args.local_rank == 1 or args.local_rank == 2 or args.local_rank == 3:
        args.logger.info(f"load ckpt from {osp.join(args.log_dir, args.ckpt)}")

    return ckpt['epoch'] + 1, ckpt['mae']


def save_ckpt(args, epoch, mae, model, optim, lr_scheduler=None):
    ckpt_dict = {
        'model': model.module.state_dict(),
        'optim': optim.state_dict(),
        'mae': mae,
        'epoch': epoch,
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
    mae_m, loss_m = AverageMeter(), AverageMeter()
    calc_m, read_m = AverageMeter(), AverageMeter()
    timer = Timer()
    log_step = len(dataloader) // 11
    if args.local_rank == 0 or args.local_rank == 1 or args.local_rank == 2  or args.local_rank == 3:
        args.writer.add_scalar('lr', optim.param_groups[0]['lr'], epoch)


    for step, sample in enumerate(dataloader):
        data, label = sample['data'].cuda(), sample['label'].cuda()
        read_m.add(timer.tiktok())

        optim.zero_grad()
        data = data.float()
        output = model(data).reshape(label.shape)
        loss = loss_fn(output, label)
        loss.backward()

        optim.step()

        calc_m.add(timer.tiktok())
        mae = torch.abs(output - label)
        mae_m.add(mae)
        loss_m.add(loss.item(), data.shape[0])
        # print(args.log_dir+'!!')
        if args.local_rank == 0 or args.local_rank == 1 or args.local_rank == 2  or args.local_rank == 3:
            global_step = (epoch - 1) * len(dataloader) + step
            args.writer.add_scalar('Loss/train', loss.item(), global_step)
            args.writer.add_scalar('MAE/train', mae.mean(), global_step)
            if (step + 1) % log_step == 0:
                args.logger.info(
                    f'[TRAIN] Epoch {epoch}@{step / len(dataloader) * 100:.1f}% | loss:{loss_m.value:.4f}, MAE:{mae_m.value:.4f}, read_time:{read_m.value:.3f}s, calc_time:{calc_m.value:.3f}s')

        timer.tiktok()

    # file.close()
    if lr_scheduler is not None:
        lr_scheduler.step()


@torch.no_grad()
def val_epoch(args, epoch, model, loss_fn, dataloader):
    model.eval()
    mae_m, loss_m = AverageMeter(), AverageMeter()

    for step, sample in enumerate(dataloader):
        data, label = sample['data'].cuda(), sample['label'].cuda()
        data = data.float()
        output = model(data).reshape(label.shape)
        loss = loss_fn(output, label)

        mae = torch.abs(output - label)
        mae_m.add(mae)
        loss_m.add(loss.item(), data.shape[0])

    args.writer.add_scalar('Loss/valid', loss_m.value, epoch)
    args.writer.add_scalar('MAE/valid', mae.mean(), epoch)
    args.logger.info(f'[VALID] Epoch {epoch} | loss:{loss_m.value:.4f}, MAE:{mae_m.value:.4f}')

    return mae_m.value


@torch.no_grad()
def test(args, model):
    load_ckpt(args, model)
    model.eval()

    test_set = BrainSet(mode='test')
    test_loader = DataLoader(test_set, 1, num_workers=24, shuffle=False)

    mae_m = AverageMeter()

    for step, sample in enumerate(test_loader):
        data, label = sample['data'].cuda(), sample['label'].cuda()
        data = data[0, ]
        data = data.float()
        output = model(data).reshape(label.shape)

        # save txt
        file = open(args.log_dir + '/test_result.txt', 'ab')
        output = output.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        np.savetxt(file, output, fmt='%.02f')
        np.savetxt(file, label, fmt='%.02f')
        file.close()

        mae = np.abs(output - label)
        mae_m.add(mae)

    args.logger.info(f'[TEST] ckpt:{osp.join(args.log_dir, args.ckpt)} | MAE:{mae_m.value:.4f}')


def train(args):
    train_set = NoisyBrainSet(mode='train')
    dev_set = BrainSet(mode='val')

    model = build_model(args)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, args.bs, num_workers=24, sampler=train_sampler, drop_last=True)
    dev_loader = DataLoader(dev_set, args.bs, num_workers=24, shuffle=False)

    loss_fn = nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.995)

    epoch_, mae = load_ckpt(args, model, optim, lr_scheduler)

    for epoch in range(epoch_, args.epochs + 1):
        train_epoch(args, epoch, model, loss_fn, optim, train_loader, lr_scheduler)
        val_mae = val_epoch(args, epoch, model, loss_fn, dev_loader)
        if val_mae is not None and val_mae < mae:
            save_ckpt(args, epoch, val_mae, model, optim, lr_scheduler)
            mae = val_mae

    # args.log_dir = '/data/ssd_2/liuyy/BrainAgeLujm/Brain_Baseline/brain-baseline/logs/resnet_region-20240311190509'
    # args.ckpt = '20240311221640-e1-11.8086.ckpt'
    test(args, model)


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed + args.local_rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5674'
    dist.init_process_group(backend='gloo', init_method='env://', rank=0,
                            world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)

    # dist.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)
    dist.barrier()

    if args.local_rank == 0 or args.local_rank == 1 or args.local_rank == 2  or args.local_rank == 3:
        args.log_dir = args.log_dir + '-' + strftime("%Y%m%d%H%M%S")
        print(args.log_dir)
        args.writer = SummaryWriter(osp.join(args.log_dir, 'tensorboard'))
        args.logger = build_logger(args.log_dir, 'train.log')

        args.logger.info(args)

    train(args)