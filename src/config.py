"""YAML configs with inheritance and command-line overrides.

A config may name a parent with `_base_: other.yaml` (path relative to the config file).
Several configs can be merged in order, e.g. `configs/cenet.yaml configs/ablation/c2_boundary.yaml`.
Overrides use dotted keys: `--opts train.epochs=5 data.root=/content/data`.
"""
import copy
from pathlib import Path

import yaml


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_update(load_yaml(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):          # YAML reads "1e-3" as a string; accept it as a number
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        node[parts[-1]] = parsed
    return cfg


def load_config(paths, overrides=None) -> dict:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    cfg = {}
    for p in paths:
        cfg = deep_update(cfg, load_yaml(p))
    # experiment name: the first config's name, extended by the stems of the extra configs
    # (e.g. "cenet" + c2_boundary.yaml -> "cenet_c2_boundary") so ablation runs never overwrite
    # the baseline run.
    name = load_yaml(paths[0]).get("experiment") or Path(paths[0]).stem
    cfg["experiment"] = "_".join([name] + [Path(p).stem for p in paths[1:]])
    cfg = apply_overrides(cfg, overrides)
    return cfg


def save_config(cfg: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
