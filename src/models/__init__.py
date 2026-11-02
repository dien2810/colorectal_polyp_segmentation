"""Model registry. Every model maps an image batch (B, 3, H, W) to logits (B, 1, H, W)."""
from .unet import UNet
from .multiresunet import MultiResUNet
from .cenet import CENet


def build_model(name: str, pretrained: bool = True, **params):
    name = name.lower()
    if name == "unet":
        return UNet(**params)
    if name == "multiresunet":
        return MultiResUNet(**params)
    if name == "cenet":
        return CENet(pretrained=pretrained, **params)
    if name == "transunet":
        from .transunet import TransUNet
        return TransUNet(pretrained=pretrained, **params)
    if name == "swinunet":
        from .swinunet import SwinUnet
        return SwinUnet(pretrained=pretrained, **params)
    if name == "respath_unet":
        from .respath_unet import ResPathUNet
        return ResPathUNet(pretrained=pretrained, **params)
    raise ValueError(f"Unknown model '{name}'. Choose from: unet, multiresunet, cenet, transunet, "
                     f"swinunet, respath_unet")


__all__ = ["build_model"]
