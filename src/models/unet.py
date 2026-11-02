"""U-Net (Ronneberger et al., MICCAI 2015).

Faithful to the original encoder-decoder layout (4 down / 4 up stages, 64-1024 channels,
transposed-convolution upsampling, skip connections). Two common modern changes are used and
reported in the report: padded convolutions (so the output has the input size) and BatchNorm.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64):
        super().__init__()
        c = [base_channels * 2 ** i for i in range(5)]  # 64, 128, 256, 512, 1024
        self.enc = nn.ModuleList([DoubleConv(in_channels, c[0])] +
                                 [DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)
