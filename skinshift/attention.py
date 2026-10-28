"""Where does the model look? Gradient-weighted attention maps on dermoscopy vs smartphone photos.

v1 showed that every method loses 25-31 accuracy points on smartphone photos, and the shortcut
probe found no reliance on border content. This asks the visual version of that question: on a
phone photo, does the model's attention move off the lesion?

Method: attention rollout weighted by gradients (after Chefer et al., 2021). In each layer the
attention map is multiplied by its gradient with respect to the predicted class, negative values
are dropped and heads are averaged; layers are combined as R <- R + A @ R, and the CLS token's row
gives one weight per 16x16 patch (a 14x14 grid). Plain attention rollout was tried first and came
out almost uniform (normalised entropy 0.99), so it is not used.

There are no lesion masks for these images, so "on the lesion" is approximated by the central 8x8
patches, where the lesion sits in most images of both datasets (see the figure). Uniform attention
would put 32.7% of its weight there.

    python -m skinshift.attention --seed 42      # one seed -> reports/attention_seed42.json + figure
    python -m skinshift.attention --all-seeds    # seeds 0, 1, 2, 3, 42 -> reports/attention.json
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .config import SEED, Paths
from .data import CLASS_NAMES

SEEDS = [0, 1, 2, 3, 42]
GRID = 14
MARGIN = 3  # central region = patches 3..10 on each axis (8x8 of 14x14)


def central_mask(grid: int = GRID, margin: int = MARGIN) -> np.ndarray:
    m = np.zeros((grid, grid), dtype=bool)
    m[margin:grid - margin, margin:grid - margin] = True
    return m.ravel()


def gradient_rollout(attentions: Sequence[torch.Tensor], grads: Sequence[torch.Tensor]) -> torch.Tensor:
    """Per-patch relevance of the CLS token, (batch, tokens - 1), each row summing to 1.

    `attentions` and `grads` are per-layer tensors of shape (batch, heads, tokens, tokens)."""
    b, _, t, _ = attentions[0].shape
    eye = torch.eye(t, device=attentions[0].device)
    r = eye.expand(b, t, t).clone()
    for a, g in zip(attentions, grads):
        cam = (g * a).clamp(min=0).mean(dim=1)
        r = r + cam @ r
    m = r[:, 0, 1:] + 1e-12  # no positive evidence anywhere -> uniform, not 0/0
    return m / m.sum(dim=-1, keepdim=True)


def central_mass(maps: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    return maps[:, central_mask() if mask is None else mask].sum(axis=1)


def spread(maps: np.ndarray) -> np.ndarray:
    """Normalised entropy: 1 = attention spread evenly over all patches, 0 = all on one patch."""
    return -(maps * np.log(maps + 1e-12)).sum(axis=1) / np.log(maps.shape[1])


def relevance_maps(model, loader, dev) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maps, logits, labels and the (normalised) input pixels for every image in `loader`."""
    model.eval()
    maps, logits, labels, pixels = [], [], [], []
    for px, y in loader:
        x = px.to(dev).requires_grad_(True)  # frozen weights, so the graph has to start at the input
        out = model(pixel_values=x, output_attentions=True)
        for a in out.attentions:
            a.retain_grad()
        pred = out.logits.argmax(dim=1, keepdim=True)
        out.logits.gather(1, pred).sum().backward()
        m = gradient_rollout(out.attentions, [a.grad for a in out.attentions])
        maps.append(m.detach().cpu().numpy())
        logits.append(out.logits.detach().cpu().numpy())
        labels.append(y.numpy())
        pixels.append(px.numpy())
        model.zero_grad(set_to_none=True)
    return tuple(np.concatenate(v) for v in (maps, logits, labels, pixels))


def summarise_split(maps: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> Dict:
    correct = logits.argmax(axis=1) == labels
    cm = central_mass(maps)
    return {"n": int(len(labels)), "accuracy": float(correct.mean()),
            "central_mass": float(cm.mean()), "spread": float(spread(maps).mean()),
            "central_mass_correct": float(cm[correct].mean()),
            "central_mass_wrong": float(cm[~correct].mean()) if (~correct).any() else float("nan")}


def plot_examples(path, id_split, ood_split, processor) -> None:
    """One image per class from each domain (the first in loader order), with its relevance map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    mean = np.array(processor.image_mean)[:, None, None]
    std = np.array(processor.image_std)[:, None, None]
    fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(2.6 * len(CLASS_NAMES), 6.4))
    for row, (name, (maps, logits, labels, pixels)) in enumerate(
            (("Dermoscopy (HAM10000)", id_split), ("Smartphone (PAD-UFES-20)", ood_split))):
        for col, cls in enumerate(CLASS_NAMES):
            ax = axes[row, col]
            ax.axis("off")
            idx = np.flatnonzero(labels == col)
            if not len(idx):
                continue
            i = idx[0]
            img = np.clip(pixels[i] * std + mean, 0, 1).transpose(1, 2, 0)
            heat = zoom(maps[i].reshape(GRID, GRID), img.shape[0] / GRID, order=1)
            ax.imshow(img)
            ax.imshow(heat, cmap="jet", alpha=0.45)
            pred = CLASS_NAMES[int(logits[i].argmax())]
            ax.set_title(f"{cls}\npredicted {pred} {'(right)' if pred == cls else '(wrong)'}", fontsize=8)
        axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes, rotation=90,
                          ha="right", va="center", fontsize=9)
    fig.suptitle("Where the LoRA model looks (gradient-weighted attention, seed 42)", fontsize=10)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96), h_pad=2.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_seed(seed: int, figure: bool = False) -> Dict:
    """Train LoRA for one seed (same protocol as train.py) and compare relevance maps ID vs OOD."""
    from .train import fit_model  # heavy import, only needed when actually training

    fitted = fit_model("lora", seed=seed)
    model, dev = fitted["model"].merge_and_unload(), fitted["device"]
    model.set_attn_implementation("eager")  # the default SDPA kernel does not return attention weights
    id_split = relevance_maps(model, fitted["id_test_dl"], dev)
    ood_split = relevance_maps(model, fitted["ood_dl"], dev)
    cm_id, cm_ood = central_mass(id_split[0]), central_mass(ood_split[0])
    result = {
        "seed": seed,
        "uniform_central_mass": float(central_mask().mean()),
        "id_test": summarise_split(*id_split[:3]),
        "ood": summarise_split(*ood_split[:3]),
        # 0.5 = attention placement does not separate the two domains at all
        "auroc_less_central_is_ood": float(roc_auc_score(np.r_[np.zeros(len(cm_id)), np.ones(len(cm_ood))],
                                                         -np.r_[cm_id, cm_ood])),
    }
    if figure:
        paths = fitted["paths"]
        paths.figures.mkdir(parents=True, exist_ok=True)
        plot_examples(paths.figures / "fig_attention_maps.png", id_split, ood_split, fitted["processor"])
    return result


def summarise(runs: List[Dict]) -> Dict:
    def stat(get):
        v = np.array([get(r) for r in runs], dtype=float)
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}

    out: Dict = {"seeds": [r["seed"] for r in runs], "uniform_central_mass": runs[0]["uniform_central_mass"],
                 "auroc_less_central_is_ood": stat(lambda r: r["auroc_less_central_is_ood"])}
    for split in ("id_test", "ood"):
        out[split] = {k: stat(lambda r, k=k, s=split: r[s][k]) for k in runs[0][split] if k != "n"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--all-seeds", action="store_true")
    args = ap.parse_args()
    paths = Paths()
    paths.reports.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (SEEDS if args.all_seeds else [args.seed]):
        r = run_seed(seed, figure=(seed == SEED))
        (paths.reports / f"attention_seed{seed}.json").write_text(json.dumps(r, indent=2))
        runs.append(r)
        print(json.dumps(r, indent=2))
    if args.all_seeds:
        (paths.reports / "attention.json").write_text(json.dumps(summarise(runs), indent=2))


if __name__ == "__main__":
    main()
