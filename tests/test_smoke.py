"""Fast checks that run on CPU without data or pretrained weights:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.losses import SegmentationLoss, signed_distance_map
from src.metrics import all_metrics, hd95, overlap_metrics
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,size,params", [
    ("unet", 64, {"base_channels": 8}),
    ("multiresunet", 64, {"base": 8}),
    ("cenet", 224, {}),   # RMP pools need >= 6x6 at 1/32 -> input >= 192
    ("transunet", 224, {"img_size": 224}),
    ("swinunet", 224, {"img_size": 224}),
    ("respath_unet", 64, {"encoder_name": "convnext_tiny"}),
])
def test_models_output_shape(name, size, params):
    model = build_model(name, pretrained=False, **params).eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, size, size))
    assert y.shape == (1, 1, size, size)


def test_metrics_known_values():
    gt = np.zeros((10, 10), bool)
    gt[2:6, 2:6] = True                      # 16 px
    pred = np.zeros((10, 10), bool)
    pred[2:6, 2:4] = True                    # 8 px, all inside
    m = overlap_metrics(pred, gt)
    assert m["dice"] == pytest.approx(2 * 8 / 24)
    assert m["iou"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert hd95(gt, gt) == 0.0
    assert hd95(np.zeros_like(gt), gt) == pytest.approx(np.hypot(10, 10))
    assert all_metrics(pred, gt)["polyp_fraction"] == pytest.approx(0.16)


def test_signed_distance_map_sign():
    mask = np.zeros((32, 32), bool)
    mask[8:24, 8:24] = True
    sdm = signed_distance_map(mask)
    assert sdm[16, 16] < 0 and sdm[0, 0] > 0
    assert abs(sdm[8, 16]) < 1e-6           # boundary pixel


def test_loss_with_boundary_term():
    loss = SegmentationLoss(boundary={"enabled": True, "lambda_max": 0.5, "ramp_epochs": 10})
    loss.set_epoch(5)
    assert loss.boundary_weight == pytest.approx(0.25)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = (torch.rand(2, 1, 32, 32) > 0.7).float()
    sdm = torch.from_numpy(np.stack([signed_distance_map(t[0].numpy() > 0.5) for t in target]))[:, None]
    value, parts = loss(logits, target, sdm)
    value.backward()
    assert torch.isfinite(value) and set(parts) == {"bce", "dice", "boundary"}


def test_config_merge_names_ablation_runs():
    cfg = load_config([ROOT / "configs/cenet.yaml", ROOT / "configs/ablation/c2_boundary.yaml"],
                      ["train.epochs=3"])
    assert cfg["experiment"] == "cenet_c2_boundary"
    assert cfg["loss"]["boundary"]["enabled"] is True
    assert cfg["train"]["epochs"] == 3 and cfg["model"]["name"] == "cenet"
