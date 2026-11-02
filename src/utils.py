"""Seeding, device selection, checkpoints, and small helpers shared by the scripts."""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def add_root_to_path():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic: a Colab disconnect never leaves a broken file


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def resolve_data_root(cfg) -> Path:
    root = Path(cfg["data"]["root"])
    return root if root.is_absolute() else ROOT / root


def run_dir(cfg) -> Path:
    out = Path(cfg.get("output_dir", "outputs/runs"))
    out = out if out.is_absolute() else ROOT / out
    return out / cfg["experiment"]
