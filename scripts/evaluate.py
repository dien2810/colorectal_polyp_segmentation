"""Evaluate a trained run on the five test sets at the original image resolution.

Examples
    python scripts/evaluate.py --run outputs/runs/cenet
    python scripts/evaluate.py --run outputs/runs/cenet --tta --save-pred

Writes
    outputs/metrics/per_image_<experiment>.csv   one row per test image (Dice, IoU, recall,
                                                 precision, HD95, polyp size)
    outputs/metrics/summary_<experiment>.csv     mean per test set, plus in-domain and
                                                 out-of-domain means and the gap (report Eq. 2)
    outputs/predictions/<experiment>/<set>/*.png (with --save-pred) binary masks for figures
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import IN_DOMAIN, OUT_DOMAIN, TEST_SETS, PolypDataset, read_split
from src.inference import load_run, predict_proba, to_original_size
from src.metrics import all_metrics
from src.utils import ROOT, get_device, resolve_data_root


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with config.yaml and best.pt")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--sets", nargs="+", default=TEST_SETS)
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation (overrides config)")
    ap.add_argument("--no-tta", action="store_true", help="disable TTA even if the config enables it")
    ap.add_argument("--save-pred", action="store_true", help="save binary predictions as PNG")
    ap.add_argument("--name", default=None, help="label used in output files (default: experiment)")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    return ap.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["dice", "iou", "recall", "precision", "hd95"]
    per_set = df.groupby("set", sort=False)[metrics].mean().reset_index()
    per_set["n"] = df.groupby("set", sort=False).size().values
    rows = [per_set]
    for label, sets in (("in_domain_mean", IN_DOMAIN), ("out_domain_mean", OUT_DOMAIN)):
        sub = per_set[per_set["set"].isin(sets)]
        if len(sub):
            r = sub[metrics].mean().to_frame().T     # mean of set means, as in report Eq. (2)
            r.insert(0, "set", label)
            r["n"] = sub["n"].sum()
            rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    s = out.set_index("set")
    if {"in_domain_mean", "out_domain_mean"} <= set(s.index):
        gap = s.loc["in_domain_mean", metrics] - s.loc["out_domain_mean", metrics]
        gap = gap.to_frame().T
        gap.insert(0, "set", "gap_in_minus_out")
        gap["n"] = np.nan
        out = pd.concat([out, gap], ignore_index=True)
    return out


def main():
    args = parse_args()
    device = get_device()
    cfg, model, transform = load_run(args.run, device, args.checkpoint)
    tta = (args.tta or cfg.get("eval", {}).get("tta", False)) and not args.no_tta
    thr = cfg.get("eval", {}).get("threshold", 0.5)
    name = args.name or cfg["experiment"]
    root = resolve_data_root(cfg)
    splits = Path(cfg["data"]["splits_dir"])
    splits = splits if splits.is_absolute() else ROOT / splits
    out = Path(args.out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    rows = []
    for test_set in args.sets:
        ds = PolypDataset(read_split(splits / f"test_{test_set}.txt"), root, transform, keep_original=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"].get("num_workers", 2))
        pred_dir = out / "predictions" / name / test_set
        if args.save_pred:
            pred_dir.mkdir(parents=True, exist_ok=True)
        for batch in tqdm(loader, desc=f"{name} | {test_set}", leave=False):
            gt = batch["orig_mask"][0].numpy().astype(bool)
            probs = predict_proba(model, batch["image"].to(device), tta=tta)
            pred = to_original_size(probs, gt.shape) > thr
            m = all_metrics(pred, gt)
            rows.append({"model": name, "set": test_set,
                         "domain": "in" if test_set in IN_DOMAIN else "out",
                         "image": batch["name"][0], **m})
            if args.save_pred:
                cv2.imwrite(str(pred_dir / f"{batch['name'][0]}.png"), pred.astype(np.uint8) * 255)

    df = pd.DataFrame(rows)
    per_image = out / "metrics" / f"per_image_{name}.csv"
    df.to_csv(per_image, index=False)
    summary = summarize(df)
    summary.insert(0, "model", name)
    summary.insert(1, "tta", tta)
    summary.to_csv(out / "metrics" / f"summary_{name}.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(summary.drop(columns=["model"]).to_string(index=False))
    print(f"Saved {per_image}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
