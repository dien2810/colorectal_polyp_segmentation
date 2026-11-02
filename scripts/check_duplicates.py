"""Look for near-duplicate images between the training set and the test sets (report Section 3.3).

Each image gets a 64-bit difference hash (dHash) on a 9x8 grayscale thumbnail; pairs whose
Hamming distance is at most --max-distance are reported. Small distances (0-5) usually mean the
same frame, possibly re-encoded or resized. Review the listed pairs by eye before removing any.

    python scripts/check_duplicates.py --max-distance 5
Writes outputs/eval/duplicates.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.data import TEST_SETS, read_split
from src.utils import ROOT


def dhash(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hashes(samples, root):
    return np.stack([dhash(root / img) for img, _, _ in samples])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/polyp")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--max-distance", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    splits = Path(args.splits) if Path(args.splits).is_absolute() else ROOT / args.splits

    train = read_split(splits / "train.txt") + read_split(splits / "val.txt")
    h_train = hashes(train, root)
    rows = []
    for test_set in TEST_SETS:
        test = read_split(splits / f"test_{test_set}.txt")
        h_test = hashes(test, root)
        dist = (h_test[:, None, :] != h_train[None, :, :]).sum(-1)       # (n_test, n_train)
        for i, j in zip(*np.nonzero(dist <= args.max_distance)):
            rows.append({"test_set": test_set, "test_image": test[i][0], "train_image": train[j][0],
                         "train_source": train[j][2], "hamming": int(dist[i, j])})
        n = len({r["test_image"] for r in rows if r["test_set"] == test_set})
        print(f"{test_set:<20} {len(test):4d} images, {n:4d} with a near duplicate in train/val")
    out = ROOT / "outputs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["test_set", "test_image", "train_image", "train_source", "hamming"]) \
        .sort_values(["hamming", "test_set"]).to_csv(out / "duplicates.csv", index=False)
    print(f"Saved {out / 'duplicates.csv'} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
