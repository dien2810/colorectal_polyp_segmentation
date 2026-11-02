"""Swin-Unet (Cao et al., ECCV Workshops 2022), reading list #56.

Encoder: Swin-T (patch 4, window 7, embed 96) pretrained on ImageNet-1k, loaded through timm.
Decoder: as in the paper, a mirror of the encoder built from Swin Transformer blocks, with
patch-expanding layers for x2 upsampling, skip connections fused by concatenation + linear
layer, and a final x4 patch expanding followed by a 1x1 projection.

Known deviations from the official code (state them in the report):
- the encoder keeps the full Swin-T depths (2, 2, 6, 2); the official Swin-Unet uses (2, 2, 2, 2)
  and loads only part of the same checkpoint;
- decoder blocks are randomly initialised instead of being copied from the encoder.
Input size is fixed at 224 x 224 (window 7 must divide every stage resolution).
"""
import torch
import torch.nn as nn
import timm
from timm.models.swin_transformer import SwinTransformerBlock


class PatchExpand(nn.Module):
    """(B, H, W, C) -> (B, s*H, s*W, C_out) with a linear layer and pixel rearrangement."""

    def __init__(self, dim: int, scale: int = 2, out_dim: int = None):
        super().__init__()
        self.scale = scale
        self.out_dim = out_dim if out_dim is not None else dim // 2
        self.expand = nn.Linear(dim, self.out_dim * scale * scale, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        b, h, w, _ = x.shape
        s, c = self.scale, self.out_dim
        x = self.expand(x).view(b, h, w, s, s, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * s, w * s, c)
        return self.norm(x)


class SwinStageUp(nn.Module):
    def __init__(self, dim: int, resolution: int, depth: int, num_heads: int, window_size: int = 7,
                 drop_path: float = 0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(dim=dim, input_resolution=(resolution, resolution), num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if i % 2 == 0 else window_size // 2,
                                 drop_path=drop_path)
            for i in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class SwinUnet(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True,
                 encoder_name: str = "swin_tiny_patch4_window7_224.ms_in1k", img_size: int = 224,
                 decoder_depths=(2, 2, 2), drop_path: float = 0.1):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0,
                                         img_size=img_size, drop_path_rate=drop_path)
        dims = [int(self.encoder.embed_dim * 2 ** i) for i in range(4)]       # 96, 192, 384, 768
        res = [img_size // 4 // 2 ** i for i in range(4)]                     # 56, 28, 14, 7
        heads = (3, 6, 12, 24)

        self.up0 = PatchExpand(dims[3])                                       # 7 -> 14, 768 -> 384
        self.fuse = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.ups = nn.ModuleList()
        for j, i in enumerate((2, 1, 0)):
            self.fuse.append(nn.Linear(2 * dims[i], dims[i]))
            self.stages.append(SwinStageUp(dims[i], res[i], decoder_depths[j], heads[i], drop_path=drop_path))
            self.ups.append(PatchExpand(dims[i]) if i > 0 else nn.Identity())
        self.norm_up = nn.LayerNorm(dims[0])
        self.final_up = PatchExpand(dims[0], scale=4, out_dim=dims[0])
        self.head = nn.Conv2d(dims[0], num_classes, 1, bias=False)

    def _encode(self, x):
        x = self.encoder.patch_embed(x)          # B, 56, 56, 96 (NHWC)
        if getattr(self.encoder, "absolute_pos_embed", None) is not None:
            x = x + self.encoder.absolute_pos_embed
        x = self.encoder.pos_drop(x) if hasattr(self.encoder, "pos_drop") else x
        skips = []
        for layer in self.encoder.layers:
            x = layer(x)
            skips.append(x)
        return self.encoder.norm(x), skips[:3]

    def forward(self, x):
        x, skips = self._encode(x)
        x = self.up0(x)
        for fuse, stage, up, skip in zip(self.fuse, self.stages, self.ups, reversed(skips)):
            x = fuse(torch.cat([x, skip], dim=-1))
            x = up(stage(x))
        x = self.final_up(self.norm_up(x))       # B, 224, 224, 96
        return self.head(x.permute(0, 3, 1, 2).contiguous())
