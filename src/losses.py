"""Losses: BCE + soft Dice (report Eq. 3) and the boundary loss of Kervadec et al. (Eq. 4)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def signed_distance_map(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the ground-truth boundary: negative inside the polyp, positive outside
    (Kervadec et al., 2021), divided by the image diagonal so that values do not depend on size."""
    mask = mask.astype(bool)
    diag = float(np.hypot(*mask.shape))
    if not mask.any():
        return np.ones(mask.shape, dtype=np.float32)
    if mask.all():
        return -np.ones(mask.shape, dtype=np.float32)
    outside = distance_transform_edt(~mask)
    inside = distance_transform_edt(mask) - 1.0
    sdm = outside * (~mask) - inside * mask
    return (sdm / diag).astype(np.float32)


def boundary_loss(logits: torch.Tensor, sdm: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) * sdm).mean()


class SegmentationLoss(nn.Module):
    """L = w_bce * BCE + w_dice * Dice (+ lambda(epoch) * boundary loss when enabled).
    lambda grows linearly from 0 to lambda_max over ramp_epochs, then stays constant."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, boundary: dict = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        boundary = boundary or {}
        self.use_boundary = bool(boundary.get("enabled", False))
        self.lambda_max = float(boundary.get("lambda_max", 0.5))
        self.ramp_epochs = max(1, int(boundary.get("ramp_epochs", 50)))
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @property
    def boundary_weight(self) -> float:
        return self.lambda_max * min(1.0, self.epoch / self.ramp_epochs) if self.use_boundary else 0.0

    def forward(self, logits, target, sdm=None):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_loss(logits, target)
        loss = self.bce_weight * bce + self.dice_weight * dice
        parts = {"bce": bce.detach(), "dice": dice.detach()}
        if self.use_boundary:
            if sdm is None:
                raise ValueError("Boundary loss is enabled but the batch has no 'sdm'")
            b = boundary_loss(logits, sdm)
            loss = loss + self.boundary_weight * b
            parts["boundary"] = b.detach()
        return loss, parts
