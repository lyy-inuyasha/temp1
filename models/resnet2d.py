import torch.nn as nn
import torch.nn.functional as F

class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.downsample = None
        if out_channels != in_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride), 
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        out = F.relu(x)
        return out


class Bottleneck2d(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1, expansion=4):
        super().__init__()
        mid_channels = in_channels // expansion
        out_channels = in_channels if out_channels is None else out_channels

        self.downsample = None
        if out_channels != in_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride), 
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        out = F.relu(x)
        return out


class ResNetBackbone2d(nn.Module):
    def __init__(self, in_channels=1, layers=[1,1,1,1], channels=[32,64,128,256], dropout=0, block=ResBlock2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels[0], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layers = []
        pre_chns = channels[0]
        for num_blocks, num_channels in zip(layers, channels):
            self.layers.append(ResNetBackbone2d._make_layer(block, num_blocks, pre_chns, num_channels, 2))
            pre_chns = num_channels
        self.layers = nn.Sequential(*self.layers)
        
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)
        x = self.dropout(x)
        x = self.gap(x)

        return x.reshape(bs, -1)


    @staticmethod
    def _make_layer(block, num_blocks, in_channels, out_channels, stride=1):
        blocks = []
        blocks.append(block(in_channels, out_channels, stride))
        for i in range(1, num_blocks):
            blocks.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*blocks)


class ResNet2d(nn.Module):
    def __init__(self, num_classes=1, layers=[1,1,1,1], channels=[32,64,128,256], dropout=0, block=ResBlock2d):
        super().__init__()
        self.backbone = ResNetBackbone2d(layers, channels, dropout, block)
        self.classifier = nn.Linear(channels[-1], num_classes)
    
    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        return out