"""Shared figure style for the report (static PDF figures, light background).

Each model keeps the same colour in every figure (colour follows the entity, never its rank).
Identity is never carried by colour alone: every figure also labels models by name.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated default palette, light mode).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_ORDER = ["unet", "multiresunet", "cenet", "transunet", "swinunet", "proposed"]
MODEL_LABELS = {"unet": "U-Net", "multiresunet": "MultiResUNet", "cenet": "CE-Net",
                "transunet": "TransUNet", "swinunet": "Swin-Unet", "proposed": "Proposed"}
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def model_key(name: str) -> str:
    """Map an experiment name such as 'cenet_c2_boundary' to its base model key."""
    for key in sorted(MODEL_ORDER, key=len, reverse=True):
        if name.startswith(key):
            return key
    return "proposed"


def model_color(name: str) -> str:
    return SLOTS[MODEL_ORDER.index(model_key(name)) % len(SLOTS)]


def model_label(name: str) -> str:
    key = model_key(name)
    label = MODEL_LABELS.get(key, name)
    suffix = name[len(key):].strip("_")
    return f"{label} ({suffix})" if suffix and key != "proposed" else label


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2, "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def save(fig, path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved {path}")
