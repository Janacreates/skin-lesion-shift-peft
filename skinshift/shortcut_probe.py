"""Shortcut probe (Day 6, stretch): does the model rely on border content -- rulers, ID
stickers, vignetting -- rather than the lesion itself?

Single seed (42) per method: this is a diagnostic, not a claim at the level of the 5-seed
headline results in the README, and is reported as such. Trains each method once (reusing
`fit_model` from train.py, so it is the *same* training code as the headline runs, not a
reimplementation that could quietly diverge), then evaluates the out-of-distribution test set
twice: once on the original images, once center-cropped to the middle 60% (removing the outer
40% of the frame, by area). If a model leans on border shortcuts, cropping them out should hurt
it more than it hurts a model that was already looking at the lesion.

    python -m skinshift.shortcut_probe
    python -m skinshift.shortcut_probe --fraction 0.4   # crop more aggressively
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from typing import Dict

from torch.utils.data import DataLoader

from .config import SEED
from .data import LesionImageDataset, center_crop_fraction, read_metadata
from .train import evaluate, fit_model

CROP_FRACTION = 0.6


def build_cropped_ood_loader(fitted: Dict, fraction: float, batch_size: int = 16) -> DataLoader:
    paths, processor = fitted["paths"], fitted["processor"]
    ood_rows = read_metadata(paths.ood_metadata)
    transform = partial(center_crop_fraction, fraction=fraction)
    ds = LesionImageDataset(ood_rows, paths.data, processor, image_transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_probe(method: str, seed: int = SEED, fraction: float = CROP_FRACTION, epochs: int = 5) -> Dict:
    fitted = fit_model(method, epochs=epochs, seed=seed)
    model, dev = fitted["model"], fitted["device"]

    original = evaluate(model, fitted["ood_dl"], dev)
    cropped_dl = build_cropped_ood_loader(fitted, fraction)
    cropped = evaluate(model, cropped_dl, dev)

    return {
        "method": method, "seed": seed, "crop_fraction": fraction,
        "ood_accuracy_original": original["accuracy"],
        "ood_accuracy_cropped": cropped["accuracy"],
        "accuracy_change_from_cropping": cropped["accuracy"] - original["accuracy"],
        "ood_macro_f1_original": original["macro_f1"],
        "ood_macro_f1_cropped": cropped["macro_f1"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=CROP_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from .config import Paths
    paths = Paths()
    results = []
    for method in ("linear_probe", "full", "lora"):
        r = run_probe(method, seed=args.seed, fraction=args.fraction)
        print(f"[{method}] OOD accuracy: original={r['ood_accuracy_original']:.3f} "
              f"cropped={r['ood_accuracy_cropped']:.3f} "
              f"change={r['accuracy_change_from_cropping']:+.3f}")
        results.append(r)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "shortcut_probe.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {paths.reports / 'shortcut_probe.json'}")


if __name__ == "__main__":
    main()
