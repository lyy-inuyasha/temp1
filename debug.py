from time import sleep

import torch
from torch.utils.data import DataLoader
from models.baseline import *
from models.vit import *

from dataset.brain import BrainSet, DisBrainSet
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging


if __name__ == '__main__':
    model = VIT(img_size=100, 
                patch_size=10, 
                num_classes=50, 
                dim=512, 
                depth=5, 
                heads=8, 
                head_dim=64, 
                mlp_dim=1024,
                dropout=.1).cuda()
    data = torch.randn((8, 1, 100, 100, 100)).cuda()
    output = model(data)
    print(output.shape)
    sleep(10)
