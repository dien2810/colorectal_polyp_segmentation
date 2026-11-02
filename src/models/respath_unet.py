"""Proposed model for improvement C1 (optional): a pretrained hierarchical encoder from timm
(PVTv2 or ConvNeXt) combined with the MultiResUNet decoder (Res paths on the skips and
MultiRes blocks after each upsampling).

Encoder features are taken at strides 4, 8, 16 and 32. Res path lengths are 3, 2, 1 for the
skips at strides 4, 8, 16, following MultiResUNet (longer paths for shallower features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .multiresunet import MultiResBlock, ResPath


class ResPathUNet(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True,
                 encoder_name: str = "pvt_v2_b2.in1k", decoder_channels=(256, 128, 64), alpha: float = 1.67):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True,
                                         out_indices=(0, 1, 2, 3))
        enc_ch = self.encoder.feature_info.channels()        # e.g. 64, 128, 320, 512 for pvt_v2_b2
        self.respaths = nn.ModuleList([ResPath(enc_ch[i], decoder_channels[2 - i], length=3 - i)
                                       for i in range(3)])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch = enc_ch[3]
        for j, i in enumerate((2, 1, 0)):
            u = decoder_channels[j]
            self.up.append(nn.ConvTranspose2d(ch, u, 2, stride=2))
            block = MultiResBlock(u * 2, u, alpha)
            self.dec.append(block)
            ch = block.out_channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ch, 32, 4, stride=4),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder(x)
        feats = [f if f.shape[1] == c else f.permute(0, 3, 1, 2)
                 for f, c in zip(feats, self.encoder.feature_info.channels())]
        skips = [rp(f) for rp, f in zip(self.respaths, feats[:3])]
        x = feats[3]
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        x = self.final(x)
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x
