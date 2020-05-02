import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

resnet_architectures = {
    'resnet18': (
        BasicBlock, [2, 2, 2, 2],
        'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet34': (
        BasicBlock, [3, 4, 6, 3],
        'https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
    'resnet50': (
        Bottleneck, [3, 4, 6, 3],
        'https://download.pytorch.org/models/resnet50-19c8e357.pth'),
    'resnet101': (
        Bottleneck, [3, 4, 23, 3],
        'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
    'resnet152': (
        Bottleneck, [3, 8, 36, 3],
        'https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
    'resnext50_32x4d': (
        Bottleneck, [3, 4, 6, 3],
        'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),
    'resnext101_32x8d': (
        Bottleneck, [3, 4, 23, 3],
        'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
    'wide_resnet50_2': (
        Bottleneck, [3, 4, 6, 3],
        'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'),
    'wide_resnet101_2': (
        Bottleneck, [3, 4, 23, 3],
        'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth')
}


class ChannelResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, in_channel=3,
                 fix=False, dropout=0.7, **kwargs):
        super().__init__(block, layers, num_classes=num_classes, **kwargs)
        if in_channel != 3:
            # Replace 3 in channels with the appropriate number
            m = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                          bias=False)
            self.conv1 = m
            # Perform normalization done for previous 2D convolution
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Delete last layer to be able to fix all other layers
        del self.fc
        # Add layer for zeroing some elements
        self.dp = nn.Dropout(p=dropout)
        # Fix previous parameters, only train the last layer
        if fix:
            for p in self.parameters():
                p.requires_grad_(False)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        # Insert new self.dp in the sequence
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc(x)
        return x


def channel_resnet(arch, pretrained=False, progress=True, in_channel=3, **kwargs):
    try:
        # Unpack tuple (frame_length, num_channels)
        frame_length, num_channels = in_channel
    except TypeError:
        frame_length, num_channels = (1, in_channel)
    params = resnet_architectures[arch]
    model = ChannelResNet(params[0], params[1],
                          in_channel=frame_length*num_channels, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(params[2], progress=progress)
        if num_channels != 3 or frame_length != 1:
            k = 'conv1.weight'
            weight = pretrained_dict[k]
            if num_channels == 3:
                # Repeat weights for stacked frames
                channel_weight = weight.repeat(1, frame_length, 1, 1)
            else:
                # Use mean weights for 'flow' and 'gray'
                weight_mean = torch.mean(weight, dim=1)
                channel_weight = weight_mean.unsqueeze(1).repeat(
                    1, frame_length * num_channels, 1, 1)
            pretrained_dict[k] = channel_weight
        # Do not use pretrained weights for last layer
        state_dict = model.state_dict()
        for k in ['fc.weight', 'fc.bias']:
            pretrained_dict[k] = state_dict[k]
        model.load_state_dict(pretrained_dict)
    return model


def channel_resnet18(**kwargs):
    return channel_resnet('resnet18', **kwargs)
