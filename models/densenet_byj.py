import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from sfcn_torch import SFCN


class _DenseLayer(nn.Module):
    def __init__(
            self,
            num_input_features: int,
            growth_rate: int,
            bn_size: int,
            drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm3d
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))

        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))

        self.conv1: nn.Conv3d
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm3d
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))

        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))

        self.conv2: nn.Conv3d
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        # self.add_module('raelu', nn.ReLU(inplace=True))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
            self,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 8,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False
    ):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=5, stride=2,
                                bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            # ('relu0', nn.ReLU(inplace=True)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('conv04', nn.Conv3d(num_features, num_features, kernel_size=1,
                                                     bias=False))
        self.features.add_module('norm2', nn.BatchNorm3d(num_features))
        # self.features.add_module('pool2', nn.AvgPool3d(3))
        self.features.add_module('drop2', nn.Dropout3d(p=0.3))
        # Linear layer
        self.avg = nn.Sequential()
        self.avg.add_module('pool2', nn.AvgPool3d(3))
        self.regressor = nn.Sequential(

            nn.Flatten(),
            nn.Linear(840, 1),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.sfcn = nn.Sequential()
        self.sfcn.add_module('sfcn', SFCN())

    def forward(self, x: Tensor) -> Tensor:
        out_s = self.sfcn(x)
        features = self.features(x)
        out_d = F.relu(features, inplace=True)
        out = torch.cat((out_s, out_d), 1)
        out = self.avg(out)
        # out = F.adaptive_avg_pool3d(out, (1, 1,1))
        out = torch.flatten(out, 1)
        out = self.regressor(out)
        return out


def _densenet(
        arch: str,
        growth_rate: int,
        block_config: Tuple[int, int, int, int],
        num_init_features: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)


def densenet_cam(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _densenet('densenet_cam', 36, (4, 6, 12, 8), 64, pretrained, progress,
                     **kwargs)


