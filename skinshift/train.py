"""Train one of three methods (linear probe / full fine-tune / LoRA) and evaluate in- and
out-of-distribution.

    python -m skinshift.train --method linear_probe
    python -m skinshift.train --method full
    python -m skinshift.train --method lora

Protocol: train on HAM10000's train split only; select nothing on the OOD set; the OOD set
(PAD-UFES-20) is touched exactly once, for the final evaluation number.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from .config import Paths, SEED
from .data import CLASS_NAMES, LesionImageDataset, check_no_lesion_overlap, lesion_split, read_metadata
from .model import device, load_model_and_processor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_loaders(paths: Paths, processor, seed: int, batch_size: int = 16
                   ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, List[Dict]]:
    train_rows_all = read_metadata(paths.train_metadata)
    train_rows, val_rows, id_test_rows = lesion_split(train_rows_all, seed=seed)
    check_no_lesion_overlap(train_rows, val_rows, id_test_rows)
    ood_rows = read_metadata(paths.ood_metadata)

    mk = lambda rows, root, shuffle: DataLoader(  # noqa: E731
        LesionImageDataset(rows, root, processor), batch_size=batch_size, shuffle=shuffle, num_workers=0)
    train_dl = mk(train_rows, paths.data, True)
    val_dl = mk(val_rows, paths.data, False)
    id_test_dl = mk(id_test_rows, paths.data, False)
    ood_dl = mk(ood_rows, paths.data, False)
    print(f"train={len(train_rows)} val={len(val_rows)} id_test={len(id_test_rows)} ood={len(ood_rows)}")
    return train_dl, val_dl, id_test_dl, ood_dl, ood_rows


def apply_method(model, method: str):
    """Return (model, trainable_param_count, total_param_count) for the requested method."""
    total = sum(p.numel() for p in model.parameters())
    if method == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif method == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif method == "lora":
        # Layer names checked directly against this transformers version's ViT implementation
        # (`model.named_modules()`) rather than assumed — it uses q_proj/k_proj/v_proj/o_proj,
        # not the older query/key/value naming some ViT/BERT variants use.
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "v_proj"], modules_to_save=["classifier"])
        model = get_peft_model(model, cfg)
    else:
        raise ValueError(method)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable, total


@torch.no_grad()
def evaluate(model, loader, dev) -> Dict:
    model.eval()
    all_probs, all_labels = [], []
    for px, y in loader:
        px = px.to(dev)
        logits = model(pixel_values=px).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean())
    # macro F1 without sklearn, to keep this function dependency-light and testable
    f1s = []
    for c in range(probs.shape[1]):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    conf = probs.max(axis=1)
    ece = expected_calibration_error(conf, preds == labels)
    return {"n": len(labels), "accuracy": acc, "macro_f1": float(np.mean(f1s)), "ece": ece,
            "probs": probs, "labels": labels, "preds": preds}


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def train_one_epoch(model, loader, opt, dev) -> float:
    model.train()
    total_loss = 0.0
    for px, y in loader:
        px, y = px.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(pixel_values=px, labels=y)
        out.loss.backward()
        opt.step()
        total_loss += float(out.loss.detach()) * len(y)
    return total_loss / len(loader.dataset)


def fit_model(method: str, epochs: int = 5, lr: float = None, seed: int = SEED) -> Dict:
    """Train one method end to end and return everything needed to evaluate it further:
    the trained (best-val-checkpoint) model, its device/processor, the standard loaders, and
    bookkeeping (trainable params, elapsed time, history). Shared by `run()` (the headline
    train+eval+save path) and `shortcut_probe.py` (which needs the live model for an extra,
    non-standard evaluation rather than just the saved metrics)."""
    set_seed(seed)
    paths = Paths()
    dev = device()
    model, processor = load_model_and_processor(num_labels=len(CLASS_NAMES))
    model, n_trainable, n_total = apply_method(model, method)
    model.to(dev)

    default_lr = {"linear_probe": 1e-3, "full": 2e-5, "lora": 1e-3}[method]
    lr = lr or default_lr
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    train_dl, val_dl, id_test_dl, ood_dl, ood_rows = build_loaders(paths, processor, seed)

    t0 = time.time()
    best_val_acc, best_state = -1.0, None
    history = []
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_dl, opt, dev)
        val = evaluate(model, val_dl, dev)
        history.append({"epoch": epoch, "train_loss": loss, "val_accuracy": val["accuracy"]})
        print(f"[{method}] epoch {epoch}: loss={loss:.3f} val_acc={val['accuracy']:.3f}")
        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model, "device": dev, "processor": processor, "paths": paths,
        "train_dl": train_dl, "val_dl": val_dl, "id_test_dl": id_test_dl, "ood_dl": ood_dl,
        "n_trainable": n_trainable, "n_total": n_total, "lr": lr,
        "train_seconds": time.time() - t0, "history": history,
    }


def run(method: str, epochs: int = 5, lr: float = None, seed: int = SEED) -> Dict:
    fitted = fit_model(method, epochs=epochs, lr=lr, seed=seed)
    model, dev, paths = fitted["model"], fitted["device"], fitted["paths"]

    id_test = evaluate(model, fitted["id_test_dl"], dev)
    ood_test = evaluate(model, fitted["ood_dl"], dev)

    result = {
        "method": method, "seed": seed, "epochs": epochs, "lr": fitted["lr"],
        "trainable_params": fitted["n_trainable"], "total_params": fitted["n_total"],
        "trainable_pct": 100 * fitted["n_trainable"] / fitted["n_total"],
        "train_seconds": fitted["train_seconds"], "history": fitted["history"],
        "id_test": {k: v for k, v in id_test.items() if k not in ("probs", "labels", "preds")},
        "ood_test": {k: v for k, v in ood_test.items() if k not in ("probs", "labels", "preds")},
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    np.savez(paths.reports / f"predictions_{method}_seed{seed}.npz",
              id_probs=id_test["probs"], id_labels=id_test["labels"],
              ood_probs=ood_test["probs"], ood_labels=ood_test["labels"])
    (paths.reports / f"result_{method}_seed{seed}.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["linear_probe", "full", "lora"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    r = run(args.method, epochs=args.epochs, lr=args.lr, seed=args.seed)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2, default=float))


if __name__ == "__main__":
    main()
