"""Statistics for the report tables (Section 6): Wilcoxon signed-rank tests on paired per-image
Dice, Friedman test with Nemenyi critical difference, bootstrap 95% CIs, generalization gap.

Reads outputs/metrics/per_image_<model>.csv written by evaluate.py.

Main comparison (Tables 5 and 6):
    python scripts/stats_tests.py --models unet multiresunet cenet transunet swinunet --reference cenet
Ablation (Table 7):
    python scripts/stats_tests.py --ablation cenet cenet_c3_domain_aug cenet_c2_boundary cenet_c2_c3

Writes CSV files and LaTeX table rows (ready to paste into the report) to outputs/tables/.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from src.data import IN_DOMAIN, OUT_DOMAIN, TEST_SETS
from src.plotting import model_label
from src.utils import ROOT

# Critical values q_0.05 of the Nemenyi test (Demšar, JMLR 2006, Table 5a), k = 2..10 models.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load(models, metrics_dir):
    frames = []
    for m in models:
        f = metrics_dir / f"per_image_{m}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: run scripts/evaluate.py for '{m}' first")
        frames.append(pd.read_csv(f).assign(model=m))
    return pd.concat(frames, ignore_index=True)


def paired(df, a, b, metric="dice", sets=None):
    sub = df if sets is None else df[df["set"].isin(sets)]
    wide = sub.pivot_table(index=["set", "image"], columns="model", values=metric)
    wide = wide[[a, b]].dropna()
    return wide[a].to_numpy(), wide[b].to_numpy()


def wilcoxon_p(x, y):
    d = x - y
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)


def mean_of_set_means(df, model, sets, metric="dice"):
    sub = df[(df["model"] == model) & (df["set"].isin(sets))]
    return sub.groupby("set")[metric].mean().mean()


def bootstrap_ci(df, model, sets, metric="dice", n_boot=2000, seed=42):
    """95% CI of the mean of set means, resampling images within each set."""
    rng = np.random.default_rng(seed)
    per_set = []
    for s in sets:
        v = df[(df["model"] == model) & (df["set"] == s)][metric].to_numpy()
        if len(v):
            idx = rng.integers(0, len(v), size=(n_boot, len(v)))
            per_set.append(v[idx].mean(axis=1))
    boots = np.mean(per_set, axis=0)
    return np.percentile(boots, [2.5, 97.5])


def friedman_nemenyi(df, models, sets, metric="dice"):
    sub = df[df["set"].isin(sets)]
    wide = sub.pivot_table(index=["set", "image"], columns="model", values=metric)[models].dropna()
    if len(models) < 3:
        return None
    stat, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in models])
    ranks = wide.rank(axis=1, ascending=False).mean()       # rank 1 = best Dice
    k, n = len(models), len(wide)
    cd = NEMENYI_Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * n))
    return {"chi2": stat, "p": p, "n_images": n, "avg_rank": ranks.to_dict(), "critical_difference": cd}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fmt(x, nd=3, comma=False):
    s = "--" if pd.isna(x) else f"{x:.{nd}f}"
    return s.replace(".", "{,}") if comma else s


def fmt_p(p, comma=False):
    s = "$<0.001$" if p < 0.001 else f"{p:.3f}"
    return s.replace(".", "{,}") if comma else s


def main_comparison(df, models, reference, out, comma):
    rows, tex5, tex6 = [], [], []
    for m in models:
        r = {"model": m}
        for s in TEST_SETS:
            r[s] = df[(df["model"] == m) & (df["set"] == s)]["dice"].mean()
        r["mdice_in"] = mean_of_set_means(df, m, IN_DOMAIN)
        r["mdice_out"] = mean_of_set_means(df, m, OUT_DOMAIN)
        r["ci_out_low"], r["ci_out_high"] = bootstrap_ci(df, m, OUT_DOMAIN)
        r["gap"] = r["mdice_in"] - r["mdice_out"]
        r["recall_out"] = mean_of_set_means(df, m, OUT_DOMAIN, "recall")
        r["hd95_out"] = mean_of_set_means(df, m, OUT_DOMAIN, "hd95")
        if reference and m != reference:
            r["p_all_vs_ref"] = wilcoxon_p(*paired(df, m, reference))
            r["p_out_vs_ref"] = wilcoxon_p(*paired(df, m, reference, sets=OUT_DOMAIN))
        rows.append(r)
        p = fmt_p(r["p_all_vs_ref"], comma) if "p_all_vs_ref" in r else "--"
        tex5.append(f"{model_label(m):<14} & " + " & ".join(fmt(r[s], comma=comma) for s in TEST_SETS)
                    + f" & {p} \\\\")
        tex6.append(f"{model_label(m):<14} & {fmt(r['mdice_in'], comma=comma)} & {fmt(r['mdice_out'], comma=comma)} "
                    f"[{fmt(r['ci_out_low'], comma=comma)}, {fmt(r['ci_out_high'], comma=comma)}] & "
                    f"{fmt(r['gap'], comma=comma)} & {fmt(r['recall_out'], comma=comma)} & "
                    f"{fmt(r['hd95_out'], 1, comma)} \\\\")
    table = pd.DataFrame(rows)
    table.to_csv(out / "main_comparison.csv", index=False)
    (out / "table5_main_rows.tex").write_text("\n".join(tex5) + "\n", encoding="utf-8")
    (out / "table6_ood_rows.tex").write_text("\n".join(tex6) + "\n", encoding="utf-8")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(table.to_string(index=False))

    fr = friedman_nemenyi(df, models, TEST_SETS)
    if fr:
        print(f"\nFriedman (all test images, n={fr['n_images']}): chi2={fr['chi2']:.2f}, p={fr['p']:.2e}")
        print("Average rank (1 = best):", {k: round(v, 2) for k, v in sorted(fr['avg_rank'].items(), key=lambda x: x[1])})
        print(f"Nemenyi critical difference (alpha=0.05): {fr['critical_difference']:.3f}")
        pd.DataFrame([{"model": k, "avg_rank": v} for k, v in fr["avg_rank"].items()]) \
            .assign(critical_difference=fr["critical_difference"], friedman_p=fr["p"]) \
            .to_csv(out / "friedman_nemenyi.csv", index=False)

    pairs = list(combinations(models, 2))
    if pairs:
        raw = np.array([wilcoxon_p(*paired(df, a, b)) for a, b in pairs])
        pd.DataFrame({"model_a": [a for a, _ in pairs], "model_b": [b for _, b in pairs],
                      "p_raw": raw, "p_holm": holm(raw)}).to_csv(out / "pairwise_wilcoxon.csv", index=False)


def ablation(df, names, out, comma):
    base = names[0]
    rows, tex = [], []
    for m in names:
        r = {"config": m, "mdice_in": mean_of_set_means(df, m, IN_DOMAIN),
             "mdice_out": mean_of_set_means(df, m, OUT_DOMAIN),
             "hd95_out": mean_of_set_means(df, m, OUT_DOMAIN, "hd95")}
        r["gap"] = r["mdice_in"] - r["mdice_out"]
        r["p_out_vs_base"] = np.nan if m == base else wilcoxon_p(*paired(df, m, base, sets=OUT_DOMAIN))
        rows.append(r)
        p = "--" if m == base else fmt_p(r["p_out_vs_base"], comma)
        tex.append(f"{model_label(m):<30} & {fmt(r['mdice_in'], comma=comma)} & {fmt(r['mdice_out'], comma=comma)} & "
                   f"{fmt(r['gap'], comma=comma)} & {fmt(r['hd95_out'], 1, comma)} & {p} \\\\")
    table = pd.DataFrame(rows)
    table.to_csv(out / "ablation.csv", index=False)
    (out / "table7_ablation_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(table.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[], help="experiments for Tables 5-6")
    ap.add_argument("--reference", default=None, help="model for the p-value column (e.g. best CNN)")
    ap.add_argument("--ablation", nargs="*", default=[], help="baseline first, then ablation runs (Table 7)")
    ap.add_argument("--metrics-dir", default=str(ROOT / "outputs" / "metrics"))
    ap.add_argument("--vi", action="store_true", help="decimal commas in LaTeX rows (Vietnamese report)")
    args = ap.parse_args()
    metrics_dir = Path(args.metrics_dir)
    out = metrics_dir.parent / "tables"
    out.mkdir(parents=True, exist_ok=True)
    if args.models:
        main_comparison(load(args.models, metrics_dir), args.models, args.reference, out, args.vi)
    if args.ablation:
        print()
        ablation(load(args.ablation, metrics_dir), args.ablation, out, args.vi)
    if not args.models and not args.ablation:
        print("Nothing to do: pass --models and/or --ablation")
    else:
        print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
