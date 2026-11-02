"""Model size and inference speed (report Table 8): parameters, FLOPs, frames per second at
batch size 1 and the model's input size. FLOPs are counted with torch.utils.flop_counter
(one multiply-add = 2 FLOPs).

    python scripts/benchmark.py --configs configs/unet.yaml configs/multiresunet.yaml \
        configs/cenet.yaml configs/transunet.yaml configs/swinunet.yaml
Writes outputs/metrics/efficiency.csv and outputs/tables/table8_cost_rows.tex.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.config import load_config
from src.models import build_model
from src.plotting import model_label
from src.utils import ROOT, count_parameters, get_device


@torch.no_grad()
def measure(model, size, device, warmup=10, iters=50):
    x = torch.randn(1, 3, size, size, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x)
    flops = fc.get_total_flops()
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return flops, iters / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    device = get_device()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    rows = []
    for path in args.configs:
        cfg = load_config(path)
        model = build_model(cfg["model"]["name"], pretrained=False, **cfg["model"].get("params", {})).to(device).eval()
        size = cfg["data"]["img_size"]
        flops, fps = measure(model, size, device, iters=args.iters)
        rows.append({"model": cfg["experiment"], "input_size": size, "params_M": count_parameters(model) / 1e6,
                     "gflops": flops / 1e9, "fps": fps, "device": gpu})
        print(f"{cfg['experiment']:<15} {size}px  {rows[-1]['params_M']:7.2f} M  {rows[-1]['gflops']:7.1f} GFLOPs  {fps:7.1f} FPS")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics" / "efficiency.csv", index=False)
    tex = [f"{model_label(r['model']):<14} & {r['params_M']:.2f} & {r['gflops']:.1f} & {r['fps']:.1f} \\\\" for r in rows]
    (ROOT / "outputs" / "tables" / "table8_cost_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Device: {gpu}. Saved outputs/metrics/efficiency.csv")


if __name__ == "__main__":
    main()
