"""Train one model configuration.

Examples
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/cenet.yaml configs/ablation/c2_boundary.yaml
    python scripts/train.py --config configs/transunet.yaml --opts train.epochs=5 output_dir=/content/drive/MyDrive/polyp_runs

The run directory (<output_dir>/<experiment>/) receives config.yaml, history.csv, last.pt
(saved every epoch, used to resume) and best.pt (highest validation Dice). Re-running the same
command resumes from last.pt automatically; pass --restart to start over.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, save_config
from src.data import PolypDataset, read_split
from src.losses import SegmentationLoss
from src.metrics import batch_dice
from src.models import build_model
from src.transforms import build_eval_transform, build_train_transform
from src.utils import (count_parameters, get_device, load_checkpoint, resolve_data_root, run_dir,
                       save_checkpoint, seed_worker, set_seed, ROOT)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, help="one or more YAML files, merged in order")
    ap.add_argument("--opts", nargs="*", default=[], help="overrides such as train.epochs=5")
    ap.add_argument("--restart", action="store_true", help="ignore an existing last.pt")
    return ap.parse_args()


def build_loaders(cfg, root):
    d = cfg["data"]
    splits = Path(d["splits_dir"])
    splits = splits if splits.is_absolute() else ROOT / splits
    norm = d.get("normalize", "imagenet")
    with_sdm = cfg["loss"]["boundary"].get("enabled", False)
    train_ds = PolypDataset(read_split(splits / "train.txt"), root,
                            build_train_transform(d["img_size"], cfg["augment"]["mode"], norm), with_sdm=with_sdm)
    val_ds = PolypDataset(read_split(splits / "val.txt"), root, build_eval_transform(d["img_size"], norm))
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    common = dict(num_workers=d.get("num_workers", 2), pin_memory=torch.cuda.is_available(),
                  worker_init_fn=seed_worker, persistent_workers=d.get("num_workers", 2) > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              drop_last=True, generator=g, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, **common)
    return train_loader, val_loader


def lr_lambda(cfg, steps_per_epoch):
    t = cfg["train"]
    warm = t.get("warmup_epochs", 0) * steps_per_epoch
    total = t["epochs"] * steps_per_epoch

    def f(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    scores = []
    for batch in loader:
        x, y = batch["image"].to(device), batch["mask"].to(device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(x)
        scores.append(batch_dice(logits.float(), y).cpu())
    return torch.cat(scores).mean().item()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    set_seed(cfg["seed"])
    device = get_device()
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    root = resolve_data_root(cfg)
    train_loader, val_loader = build_loaders(cfg, root)
    model = build_model(cfg["model"]["name"], pretrained=cfg["model"].get("pretrained", True),
                        **cfg["model"].get("params", {})).to(device)
    print(f"[{cfg['experiment']}] {cfg['model']['name']}: {count_parameters(model) / 1e6:.2f} M parameters, "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images, device={device}")

    t = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg, len(train_loader)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = SegmentationLoss(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"], cfg["loss"]["boundary"])

    start_epoch, best, best_epoch = 0, -1.0, -1
    last = out / "last.pt"
    if last.exists() and not args.restart:
        state = load_checkpoint(last, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, best_epoch = state["epoch"] + 1, state["best"], state["best_epoch"]
        print(f"Resumed after epoch {state['epoch'] + 1} (best val Dice {best:.4f} at epoch {best_epoch + 1})")

    history = out / "history.csv"
    if start_epoch == 0:
        history.write_text("epoch,train_loss,val_dice,lr,boundary_weight,seconds\n", encoding="utf-8")

    patience = t.get("early_stop_patience")
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        criterion.set_epoch(epoch)
        tic, total, n = time.time(), 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{t['epochs']}", leave=False):
            x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            sdm = batch["sdm"].to(device, non_blocking=True) if "sdm" in batch else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(x)
            loss, _ = criterion(logits.float(), y, sdm)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        val_dice = validate(model, val_loader, device, amp)
        lr = optimizer.param_groups[0]["lr"]
        with history.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, f"{total / n:.5f}", f"{val_dice:.5f}", f"{lr:.3e}",
                                    f"{criterion.boundary_weight:.4f}", f"{time.time() - tic:.1f}"])
        improved = val_dice > best
        if improved:
            best, best_epoch = val_dice, epoch
            save_checkpoint(out / "best.pt", model=model.state_dict(), epoch=epoch, val_dice=val_dice)
        save_checkpoint(last, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, best=best, best_epoch=best_epoch)
        print(f"epoch {epoch + 1:3d}  loss {total / n:.4f}  val Dice {val_dice:.4f}"
              f"{'  *best*' if improved else ''}  ({time.time() - tic:.0f}s)")
        if patience and epoch - best_epoch >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break

    print(f"Done. Best val Dice {best:.4f} at epoch {best_epoch + 1}. Run directory: {out}")


if __name__ == "__main__":
    main()
