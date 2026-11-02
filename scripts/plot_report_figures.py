"""Report figures from evaluation outputs.

    python scripts/plot_report_figures.py --models unet multiresunet cenet transunet swinunet \
        --reference cenet
Saves to report/figures/:
    generalization_gap.pdf  Fig. 3: in- vs out-of-domain mDice per model (left), per-image Dice on
                            ETIS (right)
    failure_cases.pdf       Fig. 4: the 3 lowest-Dice images of ETIS and CVC-ColonDB for the
                            reference model, with every model's prediction (needs evaluate.py
                            --save-pred for each model)
    training_curves.pdf     validation Dice per epoch for each run (from history.csv)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import plotting
from src.data import IN_DOMAIN, OUT_DOMAIN, SHORT_NAMES, load_image, load_mask, read_split
from src.utils import ROOT


def load_per_image(models, metrics_dir):
    return pd.concat([pd.read_csv(metrics_dir / f"per_image_{m}.csv").assign(model=m) for m in models],
                     ignore_index=True)


def fig_gap(df, models, out):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.5]})
    ends = []
    for m in models:
        sub = df[df["model"] == m]
        y_in = sub[sub["set"].isin(IN_DOMAIN)].groupby("set")["dice"].mean().mean()
        y_out = sub[sub["set"].isin(OUT_DOMAIN)].groupby("set")["dice"].mean().mean()
        c = plotting.model_color(m)
        ax1.plot([0, 1], [y_in, y_out], color=c, marker="o", markersize=5, zorder=3)
        ends.append([y_out, plotting.model_label(m)])
    # direct labels at the right end, pushed apart so that close values do not overlap
    ends.sort()
    lo, hi = ax1.get_ylim()
    gap = 0.065 * (hi - lo)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, label in ends:
        ax1.annotate(label, (1.06, y), va="center", fontsize=7.5, color=plotting.TEXT, annotation_clip=False)
    ax1.set_xticks([0, 1], ["In-domain", "Out-of-domain"])
    ax1.set_xlim(-0.2, 1.75)
    ax1.set_ylabel("mDice (mean of set means)")
    ax1.grid(axis="x", visible=False)

    etis = df[df["set"] == "ETIS-LaribPolypDB"]
    data = [etis[etis["model"] == m]["dice"].to_numpy() for m in models]
    bp = ax2.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={"color": plotting.TEXT, "linewidth": 1.2})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(plotting.model_color(m))
        patch.set_alpha(0.35)
        patch.set_edgecolor(plotting.model_color(m))
    rng = np.random.default_rng(0)
    for i, (v, m) in enumerate(zip(data, models), start=1):
        ax2.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=plotting.model_color(m),
                    alpha=0.6, linewidths=0, zorder=3)
    ax2.set_xticks(range(1, len(models) + 1), [plotting.model_label(m) for m in models], rotation=20)
    ax2.set_ylabel("Per-image Dice on ETIS")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    plotting.save(fig, out / "generalization_gap.pdf")


def overlay(image, mask, color):
    out = image.astype(float).copy()
    out[mask] = 0.55 * out[mask] + 0.45 * np.array(color, float)
    return out.astype(np.uint8)


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def fig_failure_cases(df, models, reference, root, splits, pred_root, out, n=3):
    import cv2
    import matplotlib.pyplot as plt
    cases = []
    for test_set in ("ETIS-LaribPolypDB", "CVC-ColonDB"):
        sub = df[(df["model"] == reference) & (df["set"] == test_set)].nsmallest(n, "dice")
        index = {Path(i).stem: (i, m) for i, m, _ in read_split(splits / f"test_{test_set}.txt")}
        cases += [(test_set, r["image"], index[str(r["image"])]) for _, r in sub.iterrows()]
    if not cases:
        print("No failure cases found")
        return
    cols = 2 + len(models)
    fig, axes = plt.subplots(len(cases), cols, figsize=(1.2 * cols, 1.05 * len(cases)), squeeze=False)
    for r, (test_set, name, (img_rel, mask_rel)) in enumerate(cases):
        image, gt = load_image(root / img_rel), load_mask(root / mask_rel).astype(bool)
        panels = [("Image", image), ("Ground truth", overlay(image, gt, (255, 255, 0)))]
        for m in models:
            p = pred_root / m / test_set / f"{name}.png"
            raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
            if raw is not None:
                pred = raw > 127
                d = df[(df["model"] == m) & (df["set"] == test_set) & (df["image"].astype(str) == str(name))]["dice"]
                title = f"{plotting.model_label(m)}\nDice {d.iloc[0]:.2f}" if len(d) else plotting.model_label(m)
                panels.append((title, overlay(image, pred, hex_rgb(plotting.model_color(m)))))
            else:
                panels.append((f"{plotting.model_label(m)}\n(no prediction)", np.full_like(image, 255)))
        for c, (title, im) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title if r == 0 or c >= 2 else "", fontsize=6.5, color=plotting.TEXT)
        axes[r, 0].set_ylabel(SHORT_NAMES[test_set], fontsize=7, color=plotting.TEXT)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
    plotting.save(fig, out / "failure_cases.pdf")


def fig_training_curves(models, runs_dir, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    drawn = 0
    for m in models:
        h = runs_dir / m / "history.csv"
        if not h.exists():
            continue
        hist = pd.read_csv(h)
        ax.plot(hist["epoch"], hist["val_dice"], color=plotting.model_color(m), linewidth=1.6,
                label=plotting.model_label(m))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    plotting.save(fig, out / "training_curves.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reference", default=None, help="model whose worst images are shown in Fig. 4")
    ap.add_argument("--root", default="data/polyp")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--runs", default=str(ROOT / "outputs" / "runs"))
    ap.add_argument("--outputs", default=str(ROOT / "outputs"), help="folder with metrics/ and predictions/")
    ap.add_argument("--out", default=str(ROOT / "report" / "figures"))
    args = ap.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    splits = Path(args.splits) if Path(args.splits).is_absolute() else ROOT / args.splits
    out = Path(args.out)

    plotting.setup()
    outputs = Path(args.outputs)
    df = load_per_image(args.models, outputs / "metrics")
    fig_gap(df, args.models, out)
    fig_failure_cases(df, args.models, args.reference or args.models[0], root, splits,
                      outputs / "predictions", out)
    fig_training_curves(args.models, Path(args.runs), out)


if __name__ == "__main__":
    main()
