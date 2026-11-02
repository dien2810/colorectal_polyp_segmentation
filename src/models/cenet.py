"""CE-Net (Gu et al., IEEE TMI 2019), reading list #60.

Follows the authors' reference implementation (CE_Net_): ResNet-34 encoder pretrained on
ImageNet, a dense atrous convolution (DAC) block and a residual multi-kernel pooling (RMP)
block at the bottleneck, and decoder blocks with 1x1 -> transposed 3x3 -> 1x1 convolutions
whose outputs are added to the encoder features.

Input size must be a multiple of 32 and at least 192 (the RMP block pools the 1/32 feature map
with kernels up to 6).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DACBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dilate1 = nn.Conv2d(channels, channels, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channels, channels, 3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channels, channels, 3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        d1 = F.relu(self.dilate1(x))
        d2 = F.relu(self.conv1x1(self.dilate2(x)))
        d3 = F.relu(self.conv1x1(self.dilate2(self.dilate1(x))))
        d4 = F.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        return x + d1 + d2 + d3 + d4


class RMPBlock(nn.Module):
    """Residual multi-kernel pooling: max-pooling with kernels 2, 3, 5, 6, a 1x1 conv to one
    channel each, upsampling, and concatenation with the input (C + 4 channels)."""

    def __init__(self, channels: int):
        super().__init__()
        self.pools = nn.ModuleList([nn.MaxPool2d(k, stride=k) for k in (2, 3, 5, 6)])
        self.conv = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [F.interpolate(self.conv(p(x)), size=(h, w), mode="bilinear", align_corners=False)
                for p in self.pools]
        return torch.cat(outs + [x], dim=1)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = in_ch // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CENet(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        filters = [64, 128, 256, 512]
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1, self.encoder2 = resnet.layer1, resnet.layer2
        self.encoder3, self.encoder4 = resnet.layer3, resnet.layer4
        self.dac = DACBlock(512)
        self.rmp = RMPBlock(512)
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1),
        )

    def forward(self, x):
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.rmp(self.dac(self.encoder4(e3)))
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        return self.final(d1)
