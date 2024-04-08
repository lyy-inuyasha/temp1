'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from models.senet import SELayer


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv1 = nn.Conv3d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(4*growth_rate)
        self.conv2 = nn.Conv3d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        # self.se = SELayer(growth_rate)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.se(out)
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm3d(in_planes)
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, bias=False)
        self.se = SELayer(out_planes)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.se(out)
        # out = F.avg_pool3d(out, 2)
        out = F.avg_pool3d(out, (1,2,2))
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=400, mode='single', num_experts=1):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv3d(2, num_planes, kernel_size=[3,7,7], stride=[1, 2, 2], padding=[1,3,3], bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm3d(num_planes)
        self.mode = mode
        if mode == 'single':
            self.linear = nn.Linear(num_planes, num_classes)
        elif mode == 'multi':
            self.linear = nn.ModuleList()
            for _ in range(num_experts):
                self.linear.append(nn.Linear(num_planes, num_classes))
        else:
            raise ValueError()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.adaptive_avg_pool3d(F.relu(self.bn(out)), output_size=[1,1,1])
        out = out.view(out.size(0), -1)

        if self.mode == 'single':
            out = self.linear(out)
            return out
        elif self.mode == 'multi':
            outs = list()
            for expert in self.linear:
                outs.append(expert(out).unsqueeze(1)) # shape: (B, 1, num_classes)
            out = torch.cat(outs, dim=1) # shape: (B, num_experts, num_classes)
            return out

def DenseNet121(**kwargs):
    return DenseNet(Bottleneck, [6,12,24,16],  **kwargs)

def DenseNet_Cam(**kwargs):
    return DenseNet(Bottleneck, [4, 6, 12, 8],  **kwargs)


def test_densenet():
    net = DenseNet_Cam(num_classes=1)
    x = torch.randn(400, 2, 26, 33, 23)
    y = net(Variable(x))
    print(y.shape)


if __name__ == '__main__':
    test_densenet()
