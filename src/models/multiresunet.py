"""MultiResUNet (Ibtehaz & Rahman, Neural Networks 2020), reading list #59.

Follows the authors' reference implementation:
- MultiRes block: three chained 3x3 convolutions with W/6, W/3, W/2 filters (W = alpha * U,
  alpha = 1.67), concatenated, plus a 1x1 residual shortcut.
- Res path: a chain of 3x3 convolutions with 1x1 residual shortcuts on every skip connection;
  lengths 4, 3, 2, 1 from the shallowest to the deepest skip.
"""
from typing import Optional

import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def multires_channels(u: int, alpha: float = 1.67) -> tuple:
    w = alpha * u
    return int(w * 0.167), int(w * 0.333), int(w * 0.5)


class MultiResBlock(nn.Module):
    def __init__(self, in_ch: int, u: int, alpha: float = 1.67):
        super().__init__()
        c3, c5, c7 = multires_channels(u, alpha)
        self.out_channels = c3 + c5 + c7
        self.shortcut = ConvBN(in_ch, self.out_channels, 1, act=False)
        self.conv3 = ConvBN(in_ch, c3, 3)
        self.conv5 = ConvBN(c3, c5, 3)
        self.conv7 = ConvBN(c5, c7, 3)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        a = self.conv3(x)
        b = self.conv5(a)
        c = self.conv7(b)
        out = self.bn1(torch.cat([a, b, c], dim=1))
        out = self.relu(out + self.shortcut(x))
        return self.bn2(out)


class ResPath(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, length: int):
        super().__init__()
        self.shortcuts = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(length):
            c_in = in_ch if i == 0 else out_ch
            self.shortcuts.append(ConvBN(c_in, out_ch, 1, act=False))
            self.convs.append(ConvBN(c_in, out_ch, 3))
            self.bns.append(nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for sc, conv, bn in zip(self.shortcuts, self.convs, self.bns):
            x = bn(self.relu(conv(x) + sc(x)))
        return x


class MultiResUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base: int = 32, alpha: float = 1.67):
        super().__init__()
        us = [base * 2 ** i for i in range(5)]  # 32, 64, 128, 256, 512
        self.pool = nn.MaxPool2d(2)

        self.enc = nn.ModuleList()
        self.respaths = nn.ModuleList()
        ch = in_channels
        enc_out = []
        for i, u in enumerate(us):
            block = MultiResBlock(ch, u, alpha)
            self.enc.append(block)
            ch = block.out_channels
            enc_out.append(ch)
            if i < 4:
                self.respaths.append(ResPath(ch, u, length=4 - i))

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(4)):
            self.up.append(nn.ConvTranspose2d(ch, us[i], 2, stride=2))
            block = MultiResBlock(us[i] * 2, us[i], alpha)
            self.dec.append(block)
            ch = block.out_channels
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            if i < 4:
                skips.append(self.respaths[i](x))
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)
