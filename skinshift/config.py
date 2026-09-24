"""Shared constants: the 4-class subset, source collections, and paths.

Data source: the ISIC Archive API (https://api.isic-archive.com), which hosts resized copies
of both HAM10000 and PAD-UFES-20 under one normalised diagnosis taxonomy. Verified 2026-09-22:
- Both collections use identical `diagnosis_3` strings for the classes below (checked directly
  against the API, not assumed), so no manual class-name alignment is needed.
- Per-image `copyright_license` and `attribution` are provided by the API and are recorded in
  the downloaded metadata for every image, not just the collection as a whole.
- The Mendeley-hosted PAD-UFES-20 bulk zips (~3.6GB) were NOT used; the ISIC Archive's resized
  copies (~330MB for the full 2,298-image collection, far less for the subset we take) are.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

API_BASE = "https://api.isic-archive.com/api/v2"

# ISIC Archive collection ids (checked live against /api/v2/collections/)
TRAIN_COLLECTION = 212  # HAM10000 — dermoscopic, license CC-BY-NC
OOD_COLLECTION = 406    # PAD-UFES-20 — clinical close-up (smartphone), license CC-BY

# Canonical diagnosis_3 strings, identical across both collections (verified against the API).
CLASSES = ["Melanoma, NOS", "Basal cell carcinoma", "Solar or actinic keratosis", "Nevus"]
CLASS_SHORT = {
    "Melanoma, NOS": "melanoma",
    "Basal cell carcinoma": "bcc",
    "Solar or actinic keratosis": "actinic_keratosis",
    "Nevus": "nevus",
}

# Per-class download caps. Actinic keratosis (149 total in HAM10000) and melanoma (52 total in
# PAD-UFES-20) are the bottleneck classes; caps above their availability just mean "take all".
TRAIN_PER_CLASS_CAP = 220
OOD_PER_CLASS_CAP = 150


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def train_images(self) -> Path:
        return self.raw / "ham10000"

    @property
    def ood_images(self) -> Path:
        return self.raw / "pad_ufes20"

    @property
    def train_metadata(self) -> Path:
        return self.data / "ham10000_metadata.csv"

    @property
    def ood_metadata(self) -> Path:
        return self.data / "pad_ufes20_metadata.csv"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def ensure(self) -> None:
        for p in (self.train_images, self.ood_images, self.figures):
            p.mkdir(parents=True, exist_ok=True)
