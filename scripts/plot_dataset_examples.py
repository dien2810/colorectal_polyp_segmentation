"""Report Fig. 1: one image per dataset with its ground-truth outline.

    python scripts/plot_dataset_examples.py
Saves report/figures/dataset_examples.pdf (+ .png).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src import plotting
from src.data import SHORT_NAMES, TEST_SETS, load_image, load_mask, read_split
from src.utils import ROOT


def outlined(image, mask, color=(255, 255, 0)):
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = max(2, round(max(image.shape[:2]) / 200))
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/polyp")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "report" / "figures" / "dataset_examples.pdf"))
    args = ap.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    splits = Path(args.splits) if Path(args.splits).is_absolute() else ROOT / args.splits

    plotting.setup()
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.75))
    for ax, test_set in zip(axes, TEST_SETS):
        samples = read_split(splits / f"test_{test_set}.txt")
        img_rel, mask_rel, _ = samples[rng.integers(len(samples))]
        image, mask = load_image(root / img_rel), load_mask(root / mask_rel)
        ax.imshow(outlined(image, mask))
        h, w = image.shape[:2]
        ax.set_title(f"{SHORT_NAMES[test_set]}\n{w}$\\times${h}", color=plotting.TEXT)
        ax.axis("off")
    fig.tight_layout(w_pad=0.4)
    plotting.save(fig, args.out)


if __name__ == "__main__":
    main()
