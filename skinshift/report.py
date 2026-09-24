"""Combine result_<method>_seed<N>.json / predictions_<method>_seed<N>.npz across all seeds
found on disk into a mean +/- std comparison table and figures.

    python -m skinshift.report

Note on what "seed" varies here: `seed` controls both the lesion-level train/val/id-test split
(skinshift.data.lesion_split) AND model initialisation/data order. Multi-seed results below
therefore capture sensitivity to the split as well as to training randomness, not training
randomness alone -- a broader, more honest robustness check, but described precisely so it is
not confused with a fixed-split multi-init study.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .config import Paths
from .data import CLASS_NAMES

METHODS = ["linear_probe", "full", "lora"]
COLORS = {"linear_probe": "#9aa0a6", "full": "#c62828", "lora": "#1f4e79"}


def risk_coverage(probs: np.ndarray, labels: np.ndarray, n_points: int = 20):
    """For each confidence threshold, what fraction of predictions is covered (kept), and what
    is the accuracy among those kept?"""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels
    thresholds = np.quantile(conf, np.linspace(0, 0.95, n_points))
    coverage, accuracy = [], []
    for t in thresholds:
        keep = conf >= t
        coverage.append(float(keep.mean()))
        accuracy.append(float(correct[keep].mean()) if keep.any() else float("nan"))
    return np.array(coverage), np.array(accuracy)


def load_seed_results(paths: Paths) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for m in METHODS:
        files = sorted(paths.reports.glob(f"result_{m}_seed*.json"))
        runs = [json.loads(f.read_text()) for f in files]
        if runs:
            out[m] = runs
    return out


def load_seed_predictions(paths: Paths) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for m in METHODS:
        files = sorted(paths.reports.glob(f"predictions_{m}_seed*.npz"))
        out[m] = [dict(np.load(f)) for f in files]
    return out


def _mean_std(values: List[float]) -> Dict[str, float]:
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def comparison_table(seed_results: Dict[str, List[dict]]) -> List[Dict]:
    rows = []
    for m, runs in seed_results.items():
        seeds = [r["seed"] for r in runs]
        id_acc = _mean_std([r["id_test"]["accuracy"] for r in runs])
        ood_acc = _mean_std([r["ood_test"]["accuracy"] for r in runs])
        id_f1 = _mean_std([r["id_test"]["macro_f1"] for r in runs])
        ood_f1 = _mean_std([r["ood_test"]["macro_f1"] for r in runs])
        id_ece = _mean_std([r["id_test"]["ece"] for r in runs])
        ood_ece = _mean_std([r["ood_test"]["ece"] for r in runs])
        drop = _mean_std([r["id_test"]["accuracy"] - r["ood_test"]["accuracy"] for r in runs])
        rows.append({
            "method": m, "n_seeds": len(runs), "seeds": seeds,
            "trainable_pct": round(runs[0]["trainable_pct"], 2),
            # median, not mean: one run can be a resource-contention outlier (e.g. two training
            # processes accidentally overlapping on the GPU) without being a wrong *result* --
            # median is robust to that in a way a mean of 5 points is not.
            "train_seconds_median": round(float(np.median([r["train_seconds"] for r in runs])), 1),
            "id_accuracy_mean": round(id_acc["mean"], 3), "id_accuracy_std": round(id_acc["std"], 3),
            "ood_accuracy_mean": round(ood_acc["mean"], 3), "ood_accuracy_std": round(ood_acc["std"], 3),
            "id_macro_f1_mean": round(id_f1["mean"], 3), "id_macro_f1_std": round(id_f1["std"], 3),
            "ood_macro_f1_mean": round(ood_f1["mean"], 3), "ood_macro_f1_std": round(ood_f1["std"], 3),
            "id_ece_mean": round(id_ece["mean"], 3), "id_ece_std": round(id_ece["std"], 3),
            "ood_ece_mean": round(ood_ece["mean"], 3), "ood_ece_std": round(ood_ece["std"], 3),
            "accuracy_drop_mean": round(drop["mean"], 3), "accuracy_drop_std": round(drop["std"], 3),
        })
    return rows


def welch_t_test(a: List[float], b: List[float]) -> Dict[str, float]:
    """Two-sided Welch's t-test, implemented without scipy (stdlib/numpy only)."""
    import math
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"diff": float(a.mean() - b.mean()), "t": float("nan"), "note": "zero variance in both groups"}
    t = (a.mean() - b.mean()) / se
    # Welch-Satterthwaite degrees of freedom (reported, not used for a p-value here --
    # with n=5 per group a normal approximation isn't reliable; report the effect and t only)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return {"diff": float(a.mean() - b.mean()), "t": float(t), "welch_df": float(df)}


def plot_accuracy_drop(seed_results: Dict[str, List[dict]], paths: Paths) -> None:
    methods = [m for m in METHODS if m in seed_results]
    id_acc = [_mean_std([r["id_test"]["accuracy"] for r in seed_results[m]]) for m in methods]
    ood_acc = [_mean_std([r["ood_test"]["accuracy"] for r in seed_results[m]]) for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - 0.18, [100 * a["mean"] for a in id_acc], 0.36, yerr=[100 * a["std"] for a in id_acc],
           label="in-distribution (HAM10000)", color="#9aa0a6", capsize=3)
    ax.bar(x + 0.18, [100 * a["mean"] for a in ood_acc], 0.36, yerr=[100 * a["std"] for a in ood_acc],
           label="out-of-distribution (PAD-UFES-20)", color="#1f4e79", capsize=3)
    n = len(seed_results[methods[0]])
    ax.set_xticks(x, methods)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy in- vs out-of-distribution ({n} seeds, error bars = 1 std)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "fig_accuracy_drop.png", dpi=150)
    plt.close(fig)


def plot_risk_coverage(seed_preds: Dict[str, List[dict]], paths: Paths, split: str) -> None:
    grid = np.linspace(5, 100, 20)  # common coverage grid, in %
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in METHODS:
        runs = seed_preds.get(m, [])
        if not runs:
            continue
        curves = []
        for r in runs:
            cov, acc = risk_coverage(r[f"{split}_probs"], r[f"{split}_labels"])
            # risk_coverage() walks increasing confidence thresholds, so `cov` comes back
            # monotonically DEcreasing; np.interp requires its x-array ascending, so reverse
            # both arrays first. (A direct ax.plot of the un-reversed arrays looks fine, since
            # a line plot just connects points in array order -- only interp needs this.)
            curves.append(np.interp(grid, 100 * cov[::-1], 100 * acc[::-1]))
        curves = np.array(curves)
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(grid)
        ax.plot(grid, mean, marker="o", markersize=3, label=m, color=COLORS[m])
        ax.fill_between(grid, mean - std, mean + std, color=COLORS[m], alpha=0.15)
    ax.set_xlabel("Coverage: % of predictions kept (rest abstain)")
    ax.set_ylabel("Accuracy among kept predictions (%)")
    n = len(next(iter(seed_preds.values())))
    ax.set_title(f"Risk-coverage curve ({split}, {n} seeds, band = 1 std)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(paths.figures / f"fig_risk_coverage_{split}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    paths = Paths()
    paths.figures.mkdir(parents=True, exist_ok=True)
    seed_results = load_seed_results(paths)
    if not seed_results:
        print("No result_*_seed*.json files found yet — run skinshift.train first.")
        return
    seed_preds = load_seed_predictions(paths)

    table = comparison_table(seed_results)
    for row in table:
        print(row)
    (paths.reports / "comparison_table.json").write_text(json.dumps(table, indent=2))

    if "full" in seed_results and "lora" in seed_results:
        t_ood = welch_t_test([r["ood_test"]["accuracy"] for r in seed_results["lora"]],
                              [r["ood_test"]["accuracy"] for r in seed_results["full"]])
        print("\nLoRA vs full fine-tune, OOD accuracy (Welch t-test):", t_ood)
        (paths.reports / "lora_vs_full_ood_ttest.json").write_text(json.dumps(t_ood, indent=2))

    plot_accuracy_drop(seed_results, paths)
    if seed_preds:
        plot_risk_coverage(seed_preds, paths, "id")
        plot_risk_coverage(seed_preds, paths, "ood")
    print(f"\nFigures written to {paths.figures}")


if __name__ == "__main__":
    main()
