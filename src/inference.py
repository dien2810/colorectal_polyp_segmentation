"""Prediction helpers shared by evaluation, the figure scripts and the demo."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_yaml
from .models import build_model
from .transforms import build_eval_transform
from .utils import load_checkpoint


@torch.no_grad()
def predict_proba(model, image: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """image: (B, 3, S, S) -> probabilities (B, 1, S, S). TTA averages over identity,
    horizontal flip and vertical flip (improvement C3)."""
    probs = torch.sigmoid(model(image))
    if tta:
        probs = probs + torch.sigmoid(model(image.flip(-1))).flip(-1)
        probs = probs + torch.sigmoid(model(image.flip(-2))).flip(-2)
        probs = probs / 3.0
    return probs


def to_original_size(probs: torch.Tensor, size) -> np.ndarray:
    """Bilinear resize of a (1, 1, S, S) probability map to (H, W); returns a numpy array."""
    up = F.interpolate(probs.float(), size=tuple(size), mode="bilinear", align_corners=False)
    return up[0, 0].cpu().numpy()


def load_run(run_dir, device, checkpoint: str = "best.pt"):
    """Load config and weights of a training run directory."""
    run_dir = Path(run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    model = build_model(cfg["model"]["name"], pretrained=False, **cfg["model"].get("params", {}))
    state = load_checkpoint(run_dir / checkpoint)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    transform = build_eval_transform(cfg["data"]["img_size"], cfg["data"].get("normalize", "imagenet"))
    return cfg, model, transform
