"""Per-image metrics computed on binary masks at the original resolution (report Section 6).

HD95 is implemented here with scipy so that every model is scored by the same code. Conventions:
- boundaries are the mask pixels removed by a 3x3 binary erosion;
- HD95 = max of the two directed 95th-percentile distances (pixels);
- an empty prediction on a non-empty ground truth gets the image diagonal.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(np.hypot(*gt.shape))
    bp, bg = _boundary(pred), _boundary(gt)
    dist_to_gt = distance_transform_edt(~bg)
    dist_to_pred = distance_transform_edt(~bp)
    d_pg = np.percentile(dist_to_gt[bp], 95)
    d_gp = np.percentile(dist_to_pred[bg], 95)
    return float(max(d_pg, d_gp))


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    if tp + fp + fn == 0:          # both empty
        return {"dice": 1.0, "iou": 1.0, "recall": 1.0, "precision": 1.0}
    return {
        "dice": float(2 * tp / (2 * tp + fp + fn)),
        "iou": float(tp / (tp + fp + fn)),
        "recall": float(tp / (tp + fn)) if tp + fn > 0 else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp > 0 else 0.0,
    }


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    out = overlap_metrics(pred, gt)
    out["hd95"] = hd95(pred, gt)
    out["polyp_fraction"] = float(gt.astype(bool).mean())
    return out


def batch_dice(logits, target, threshold: float = 0.5):
    """Fast Dice per image on tensors (used for validation at model resolution)."""
    import torch
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = torch.where(denom > 0, 2 * inter / denom.clamp(min=1e-8), torch.ones_like(denom))
    return dice
