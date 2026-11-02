"""Create a small synthetic dataset in the PraNet layout to test the whole pipeline in minutes
(no download, runs on CPU). Images are textured backgrounds with one reddish ellipse ("polyp").

    python scripts/make_dummy_data.py --root data/dummy
    python scripts/prepare_splits.py --root data/dummy --out data/splits_dummy
    python scripts/train.py --config configs/unet.yaml --opts data.root=data/dummy \
        data.splits_dir=data/splits_dummy train.epochs=2 train.batch_size=4 experiment=dummy_unet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.data import TEST_SETS
from src.utils import ROOT


def sample(rng, h, w, tint):
    img = rng.normal(0, 12, (h, w, 3)) + np.array(tint, float)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    mask = np.zeros((h, w), np.uint8)
    center = (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4)))
    axes = (int(rng.integers(w // 12, w // 5)), int(rng.integers(h // 12, h // 5)))
    cv2.ellipse(mask, center, axes, float(rng.integers(0, 180)), 0, 360, 255, -1)
    polyp = np.array(tint, float) * np.array([1.15, 0.8, 0.8])
    img[mask > 0] = 0.6 * img[mask > 0] + 0.4 * polyp
    return np.clip(img, 0, 255).astype(np.uint8), mask


def write(folder, items):
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "masks").mkdir(parents=True, exist_ok=True)
    for name, img, mask in items:
        cv2.imwrite(str(folder / "images" / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(folder / "masks" / f"{name}.png"), mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dummy")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=6)
    args = ap.parse_args()
    root = ROOT / args.root
    rng = np.random.default_rng(0)
    train = []
    for i in range(args.n_train):
        if i % 2:   # numeric name -> treated as CVC-ClinicDB, like the PraNet release
            train.append((str(i + 1), *sample(rng, 288, 384, (150, 90, 80))))
        else:       # alphanumeric ID -> treated as Kvasir-SEG
            train.append((f"cju{i:04d}dummy", *sample(rng, 400, 480, (170, 100, 90))))
    write(root / "TrainDataset", train)
    tints = [(170, 100, 90), (150, 90, 80), (120, 110, 100), (190, 120, 120), (140, 80, 70)]
    for k, test_set in enumerate(TEST_SETS):
        write(root / "TestDataset" / test_set,
              [(f"{test_set.lower()}_{i}", *sample(rng, 300 + 40 * k, 360 + 40 * k, tints[k]))
               for i in range(args.n_test)])
    print(f"Synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
