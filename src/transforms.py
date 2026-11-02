"""Augmentation pipelines (albumentations 2.0.x).

- "baseline": flips, mild affine (scale, rotation, shift), mild brightness/contrast.
- "strong" (improvement C3): baseline + stronger photometric changes that imitate differences
  between endoscopes and centers (colour, hue/saturation, gamma, blur, JPEG compression).
"""
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

NORMALIZATION = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # timm ViT hybrid (TransUNet) weights
}


def _norm(name: str):
    mean, std = NORMALIZATION[name]
    return [A.Normalize(mean=mean, std=std), ToTensorV2()]


def build_train_transform(size: int, mode: str = "baseline", normalize: str = "imagenet"):
    geometric = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.75, 1.25), rotate=(-15, 15), translate_percent=(-0.05, 0.05),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
    ]
    if mode == "baseline":
        photometric = [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)]
    elif mode == "strong":
        photometric = [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.RandomGamma(gamma_limit=(70, 150), p=0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 7))], p=0.3),
            A.ImageCompression(quality_range=(40, 95), p=0.3),
        ]
    else:
        raise ValueError(f"Unknown augmentation mode '{mode}' (use 'baseline' or 'strong')")
    return A.Compose(geometric + photometric + _norm(normalize))


def build_eval_transform(size: int, normalize: str = "imagenet"):
    return A.Compose([A.Resize(size, size)] + _norm(normalize))
