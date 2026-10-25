"""Shift detection: before trusting a prediction, ask whether the image looks like the training data.

The classifier loses 25-31 accuracy points on smartphone photos but stays fairly confident, so its
confidence alone is a weak warning sign. This module compares three per-image "unusualness"
scores (higher = more unusual):

- msp: 1 - max softmax probability, i.e. the model's own confidence (baseline);
- energy: -logsumexp(logits) (Liu et al., 2020) (baseline);
- mahalanobis: distance from the image's CLS embedding to the nearest class mean, measured with
  one covariance shared across classes (Lee et al., 2018), all fit on the training split.

Protocol, unchanged from train.py: fit on the training split, set each threshold on in-distribution
validation images so that 95% of them pass, and touch the smartphone (OOD) set only for the final
numbers.

    python -m skinshift.shift --seed 42      # one seed -> reports/shift_seed42.json
    python -m skinshift.shift --all-seeds    # seeds 0, 1, 2, 3, 42 -> reports/shift_detection.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from .config import SEED, Paths

SEEDS = [0, 1, 2, 3, 42]
KEEP_ID = 0.95        # each detector's threshold lets 95% of in-distribution validation images pass
KEEP_CONFIDENT = 0.80  # among images that pass, answer only the 80% most confident (on ID validation)
SCORES = ["msp", "energy", "mahalanobis"]


@dataclass
class MahalanobisDetector:
    """Class means plus one shared precision matrix, fit on training-set CLS embeddings."""

    means: np.ndarray      # (n_classes, dim)
    precision: np.ndarray  # (dim, dim)

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray) -> "MahalanobisDetector":
        classes = np.unique(labels)
        means = np.stack([features[labels == c].mean(axis=0) for c in classes])
        centered = features - means[np.searchsorted(classes, labels)]
        # Ledoit-Wolf shrinkage: a few hundred images in 192 dimensions give a noisy sample covariance.
        precision = LedoitWolf().fit(centered).precision_
        return cls(means=means, precision=precision)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance to the nearest class mean (higher = more unusual)."""
        diff = features[:, None, :] - self.means[None, :, :]               # (n, classes, dim)
        d2 = np.einsum("ncd,de,nce->nc", diff, self.precision, diff)
        return d2.min(axis=1)


def msp_score(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return 1.0 - p.max(axis=1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1)
    return -(m + np.log(np.exp(logits - m[:, None]).sum(axis=1)))


def threshold_keeping(scores: np.ndarray, keep: float = KEEP_ID) -> float:
    """The score below which a `keep` fraction of these (in-distribution) images fall."""
    return float(np.quantile(scores, keep))


@torch.no_grad()
def embed(model, loader, dev) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLS embeddings (after the final layer norm, i.e. what the classifier sees), logits, labels.

    `model` must be a plain ViTForImageClassification (merge LoRA weights first)."""
    model.eval()
    feats, logits, labels = [], [], []
    for px, y in loader:
        h = model.vit(pixel_values=px.to(dev)).last_hidden_state[:, 0]
        feats.append(h.cpu().numpy())
        logits.append(model.classifier(h).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(labels)


def all_scores(detector: MahalanobisDetector, feats: np.ndarray, logits: np.ndarray) -> Dict[str, np.ndarray]:
    return {"msp": msp_score(logits), "energy": energy_score(logits), "mahalanobis": detector.score(feats)}


def evaluate_detectors(detector: MahalanobisDetector, val: Tuple, id_test: Tuple, ood: Tuple) -> Dict:
    """AUROC and catch rate at a 95%-pass threshold for each score, plus the gated app's outcome.

    Each argument after `detector` is a (features, logits, labels) tuple from `embed`."""
    s_val, s_id, s_ood = (all_scores(detector, f, l) for f, l, _ in (val, id_test, ood))
    out: Dict = {"n_val": len(val[2]), "n_id_test": len(id_test[2]), "n_ood": len(ood[2]), "detectors": {}}
    for name in SCORES:
        thr = threshold_keeping(s_val[name])
        y = np.r_[np.zeros(len(s_id[name])), np.ones(len(s_ood[name]))]
        out["detectors"][name] = {
            "auroc": float(roc_auc_score(y, np.r_[s_id[name], s_ood[name]])),
            "ood_caught": float((s_ood[name] > thr).mean()),
            "id_test_false_alarm": float((s_id[name] > thr).mean()),
        }

    # The app's decision: refuse unfamiliar images (Mahalanobis), then answer only confident ones.
    shift_thr = threshold_keeping(s_val["mahalanobis"])
    familiar_val = s_val["mahalanobis"] <= shift_thr
    conf_thr = float(np.quantile(1 - s_val["msp"][familiar_val], 1 - KEEP_CONFIDENT))
    out["thresholds"] = {"mahalanobis": shift_thr, "confidence": conf_thr}
    for split, (f, l, labels), s in (("id_test", id_test, s_id), ("ood", ood, s_ood)):
        unfamiliar = s["mahalanobis"] > shift_thr
        answered = ~unfamiliar & (1 - s["msp"] >= conf_thr)
        correct = l.argmax(axis=1) == labels
        out[split] = {
            "accuracy_all": float(correct.mean()),
            "refused_unfamiliar": float(unfamiliar.mean()),
            "refused_not_confident": float((~unfamiliar & ~answered).mean()),
            "answered": float(answered.mean()),
            "accuracy_when_answered": float(correct[answered].mean()) if answered.any() else float("nan"),
        }
    return out


def run_seed(seed: int) -> Dict:
    """Train LoRA for one seed (same protocol as train.py), then fit and score the detectors."""
    from .train import fit_model  # heavy import, only needed when actually training

    fitted = fit_model("lora", seed=seed)
    model, dev = fitted["model"].merge_and_unload(), fitted["device"]
    train = embed(model, fitted["train_dl"], dev)
    detector = MahalanobisDetector.fit(train[0], train[2])
    result = evaluate_detectors(detector, embed(model, fitted["val_dl"], dev),
                                embed(model, fitted["id_test_dl"], dev), embed(model, fitted["ood_dl"], dev))
    result["seed"] = seed
    return result


def summarise(runs: List[Dict]) -> Dict:
    """Mean and std across seeds for every number in the per-seed results."""
    def stat(get):
        v = np.array([get(r) for r in runs], dtype=float)
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}

    summary: Dict = {"seeds": [r["seed"] for r in runs], "detectors": {}}
    for name in SCORES:
        summary["detectors"][name] = {k: stat(lambda r, k=k, n=name: r["detectors"][n][k])
                                      for k in ("auroc", "ood_caught", "id_test_false_alarm")}
    for split in ("id_test", "ood"):
        summary[split] = {k: stat(lambda r, k=k, s=split: r[s][k]) for k in runs[0][split]}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--all-seeds", action="store_true")
    args = ap.parse_args()
    paths = Paths()
    paths.reports.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (SEEDS if args.all_seeds else [args.seed]):
        r = run_seed(seed)
        (paths.reports / f"shift_seed{seed}.json").write_text(json.dumps(r, indent=2))
        runs.append(r)
        print(json.dumps({"seed": seed, **{n: r["detectors"][n] for n in SCORES}}, indent=2))
    if args.all_seeds:
        (paths.reports / "shift_detection.json").write_text(json.dumps(summarise(runs), indent=2))


if __name__ == "__main__":
    main()
