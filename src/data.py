"""Datasets and split files.

Expected raw layout (as released with PraNet):
    <root>/TrainDataset/images/*.png      <root>/TrainDataset/masks/*.png
    <root>/TestDataset/<SET>/images/*      <root>/TestDataset/<SET>/masks/*
with <SET> in TEST_SETS. Split files list one sample per line:
    <image path relative to root>\t<mask path relative to root>\t<source dataset>
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .losses import signed_distance_map

TEST_SETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
IN_DOMAIN = ["Kvasir", "CVC-ClinicDB"]
OUT_DOMAIN = ["CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
SHORT_NAMES = {"Kvasir": "Kvasir", "CVC-ClinicDB": "ClinicDB", "CVC-ColonDB": "ColonDB",
               "ETIS-LaribPolypDB": "ETIS", "CVC-300": "CVC-300"}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

Sample = Tuple[str, str, str]  # image, mask, source


def infer_train_source(filename: str) -> str:
    """In the PraNet training folder, CVC-ClinicDB images have numeric names (e.g. '12.png'),
    Kvasir-SEG images have alphanumeric IDs (e.g. 'cju0qkwl35piu0993l0dewei2.png')."""
    return "CVC-ClinicDB" if Path(filename).stem.isdigit() else "Kvasir"


def pair_images_and_masks(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs = []
    for img in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        if img.stem not in masks:
            raise FileNotFoundError(f"No mask for {img} in {mask_dir}")
        pairs.append((img, masks[img.stem]))
    return pairs


def read_split(path) -> List[Sample]:
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            img, mask, source = line.split("\t")
            samples.append((img, mask, source))
    return samples


def write_split(path, samples: List[Sample]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(f"{i}\t{m}\t{s}\n" for i, m, s in samples), encoding="utf-8")


def load_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 127).astype(np.uint8)


class PolypDataset(Dataset):
    """Returns a dict with 'image' (3,S,S float), 'mask' (1,S,S float), 'name', 'source'.
    With with_sdm=True also returns 'sdm' (1,S,S), the signed distance map used by the
    boundary loss. With keep_original=True also returns the original-size mask (numpy),
    used to score predictions at the original resolution."""

    def __init__(self, samples: List[Sample], root, transform, with_sdm: bool = False,
                 keep_original: bool = False):
        self.samples = samples
        self.root = Path(root)
        self.transform = transform
        self.with_sdm = with_sdm
        self.keep_original = keep_original

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel, mask_rel, source = self.samples[idx]
        image = load_image(self.root / img_rel)
        mask = load_mask(self.root / mask_rel)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = self.transform(image=image, mask=mask)
        m = out["mask"]
        m = m.float() if torch.is_tensor(m) else torch.from_numpy(m).float()
        item = {"image": out["image"], "mask": m.unsqueeze(0), "name": Path(img_rel).stem, "source": source}
        if self.with_sdm:
            item["sdm"] = torch.from_numpy(signed_distance_map(m.numpy() > 0.5)).unsqueeze(0)
        if self.keep_original:
            item["orig_mask"] = mask
        return item
