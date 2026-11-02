"""Build the split files from the PraNet data release (report Section 3.3, Table 2).

- train/val: the 1,450 PraNet training images, 10% of each source (Kvasir-SEG, CVC-ClinicDB)
  held out at random for validation (seed 42) -> 810 + 495 train, 90 + 55 validation.
- test_<SET>.txt: the five PraNet test sets, unchanged.

    python scripts/prepare_splits.py --root data/polyp
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TEST_SETS, infer_train_source, locate_pair_dirs, pair_images_and_masks, write_split
from src.utils import ROOT

EXPECTED = {"Kvasir": 900, "CVC-ClinicDB": 550}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/polyp")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out

    train_dir = root / "TrainDataset"
    if not train_dir.is_dir():
        present = sorted(p.name for p in root.iterdir()) if root.is_dir() else "folder does not exist"
        sys.exit(f"{train_dir} not found (contents of {root}: {present}).\n"
                 f"Run: python scripts/download_data.py --dest {root}")
    img_dir, mask_dir = locate_pair_dirs(train_dir)
    print(f"TrainDataset: images in '{img_dir.relative_to(root)}', masks in '{mask_dir.relative_to(root)}'")
    pairs = pair_images_and_masks(img_dir, mask_dir)
    by_source = {}
    for img, mask in pairs:
        by_source.setdefault(infer_train_source(img.name), []).append(
            (img.relative_to(root).as_posix(), mask.relative_to(root).as_posix()))

    rng = random.Random(args.seed)
    train, val = [], []
    for source in sorted(by_source):
        items = sorted(by_source[source])
        rng.shuffle(items)
        n_val = round(len(items) * args.val_fraction)
        val += [(i, m, source) for i, m in items[:n_val]]
        train += [(i, m, source) for i, m in items[n_val:]]
        if source in EXPECTED and len(items) != EXPECTED[source]:
            print(f"WARNING: {source} has {len(items)} training images, expected {EXPECTED[source]}. "
                  f"Check the data release and infer_train_source() in src/data.py.")
    write_split(out / "train.txt", sorted(train))
    write_split(out / "val.txt", sorted(val))

    print(f"{'split':<22}{'images':>8}   per source")
    for name, items in (("train", train), ("val", val)):
        print(f"{name:<22}{len(items):>8}   {dict(Counter(s for _, _, s in items))}")
    for test_set in TEST_SETS:
        d = root / "TestDataset" / test_set
        tp = pair_images_and_masks(*locate_pair_dirs(d))
        items = [(i.relative_to(root).as_posix(), m.relative_to(root).as_posix(), test_set) for i, m in tp]
        write_split(out / f"test_{test_set}.txt", items)
        print(f"{'test_' + test_set:<22}{len(items):>8}")
    print(f"Split files written to {out}")


if __name__ == "__main__":
    main()
