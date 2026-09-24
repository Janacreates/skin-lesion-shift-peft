"""Lesion-level splits and a torch Dataset over the downloaded images.

Splitting by lesion (not by image) matters because HAM10000 has multiple images of the same
lesion; putting two photos of one lesion in both train and test would leak information and
inflate the reported accuracy. `lesion_id` comes straight from the ISIC Archive metadata.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .config import CLASS_SHORT, Paths

CLASS_NAMES = list(CLASS_SHORT.values())  # fixed order: consistent label ids everywhere
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def read_metadata(path: Path) -> List[Dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lesion_split(rows: List[Dict], seed: int, val_frac: float = 0.15, test_frac: float = 0.15
                  ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split by lesion_id (falling back to isic_id when a lesion id is missing) so that all
    images of one lesion land in the same split."""
    rng = np.random.default_rng(seed)
    key = lambda r: r["lesion_id"] or r["isic_id"]  # noqa: E731
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    lesion_ids = list(groups)
    rng.shuffle(lesion_ids)
    n = len(lesion_ids)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_ids = set(lesion_ids[:n_test])
    val_ids = set(lesion_ids[n_test:n_test + n_val])
    train_ids = set(lesion_ids[n_test + n_val:])

    train = [r for lid in train_ids for r in groups[lid]]
    val = [r for lid in val_ids for r in groups[lid]]
    test = [r for lid in test_ids for r in groups[lid]]
    return train, val, test


def check_no_lesion_overlap(*splits: List[Dict]) -> None:
    """Raise if any lesion_id (or isic_id fallback) appears in more than one split."""
    seen: Dict[str, int] = {}
    for i, split in enumerate(splits):
        for r in split:
            k = r["lesion_id"] or r["isic_id"]
            if k in seen and seen[k] != i:
                raise ValueError(f"lesion/image {k!r} appears in more than one split")
            seen[k] = i


def center_crop_fraction(img: Image.Image, fraction: float) -> Image.Image:
    """Crop to the central `fraction` of both width and height (e.g. 0.6 keeps the middle 60%),
    without resizing back -- the model's own processor resizes whatever it's given. Used by the
    shortcut probe: if a model relies on border content (rulers, stickers, vignetting) rather
    than the lesion, removing the border should hurt it more than it hurts a model that doesn't."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    w, h = img.size
    cw, ch = w * fraction, h * fraction
    left, top = (w - cw) / 2, (h - ch) / 2
    return img.crop((left, top, left + cw, top + ch))


class LesionImageDataset(Dataset):
    """rows: metadata dicts from read_metadata(); root: the dataset's data/ directory
    (so each row's relative `path` resolves correctly). `image_transform`, if given, is applied
    to the PIL image (e.g. center_crop_fraction) before the model's own processor runs."""

    def __init__(self, rows: List[Dict], root: Path, processor, image_transform=None):
        self.rows = rows
        self.root = root
        self.processor = processor
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.root / row["path"]).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"][0]
        label = LABEL_TO_ID[row["class_short"]]
        return pixel_values, label
