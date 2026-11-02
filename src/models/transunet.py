"""TransUNet (Chen et al., 2021), reading list #57.

Encoder: the R50+ViT-B/16 hybrid of the original paper. A ResNet-50 (v2, StdConv + GroupNorm,
layers 3-4-9) produces 1/16-resolution features that are projected to 768-d tokens and passed
through 12 Transformer layers. The pretrained weights are the same ImageNet-21k R50+ViT-B/16
checkpoint used by TransUNet, loaded through timm ("vit_base_r50_s16_224.orig_in21k").

Decoder: the cascaded upsampler (CUP) of the official code: a 3x3 conv (768 -> 512), then four
x2 upsampling blocks with channels 256, 128, 64, 16 and skip connections from the ResNet
features at 1/8 (512 ch), 1/4 (256 ch) and 1/2 (64 ch), and a 3x3 segmentation head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import resample_abs_pos_embed


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__(nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int = 0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch)
        self.conv2 = Conv2dReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class TransUNet(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True,
                 vit_name: str = "vit_base_r50_s16_224.orig_in21k", img_size: int = 224,
                 decoder_channels=(256, 128, 64, 16)):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.backbone = self.vit.patch_embed.backbone   # ResNetV2 (stem + 3 stages)
        self.proj = self.vit.patch_embed.proj           # 1x1 conv 1024 -> 768
        embed_dim = self.vit.embed_dim
        skip_channels = (512, 256, 64, 0)

        self.conv_more = Conv2dReLU(embed_dim, 512)
        in_chs = (512,) + tuple(decoder_channels[:-1])
        self.decoder = nn.ModuleList([DecoderBlock(i, o, s) for i, o, s
                                      in zip(in_chs, decoder_channels, skip_channels)])
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, 3, padding=1)

    def _encode(self, x):
        stem = self.backbone.stem
        x = stem.norm(stem.conv(x))           # 1/2, 64 ch  (conv -> GN -> ReLU)
        f_half = x
        x = stem.pool(x)                      # 1/4
        feats = []
        for stage in self.backbone.stages:    # 1/4 (256), 1/8 (512), 1/16 (1024)
            x = stage(x)
            feats.append(x)
        x = self.backbone.norm(x)
        return x, [feats[1], feats[0], f_half]

    def forward(self, x):
        h, w = x.shape[-2:]
        x, skips = self._encode(x)
        x = self.proj(x)                                  # B, 768, H/16, W/16
        gh, gw = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        n_prefix = self.vit.num_prefix_tokens
        pos = self.vit.pos_embed[:, n_prefix:] if not getattr(self.vit, "no_embed_class", False) else self.vit.pos_embed
        old = int(round(pos.shape[1] ** 0.5))
        if (gh, gw) != (old, old):
            pos = resample_abs_pos_embed(pos, new_size=(gh, gw), old_size=(old, old), num_prefix_tokens=0)
        tokens = self.vit.pos_drop(tokens + pos)
        tokens = self.vit.norm_pre(tokens)
        tokens = self.vit.blocks(tokens)
        tokens = self.vit.norm(tokens)
        x = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], gh, gw)
        x = self.conv_more(x)
        for i, block in enumerate(self.decoder):
            x = block(x, skips[i] if i < len(skips) else None)
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(x)
