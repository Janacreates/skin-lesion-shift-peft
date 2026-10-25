"""Package the LoRA model for the demo app: train it, merge the adapter into the ViT, export to ONNX
(fp32, then int8), fit the shift detector on the exported model's own features, and benchmark
size, speed and agreement against PyTorch.

    python -m skinshift.export   # writes deploy/model/, deploy/examples/, reports/deploy_benchmark.json

The detector is fit on the int8 model's features, because those are what the app computes.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from deploy.core import DISPLAY_NAMES, SkinClassifier, decide, save_detector
from .config import SEED, Paths
from .data import CLASS_NAMES
from .shift import KEEP_CONFIDENT, MahalanobisDetector, evaluate_detectors, msp_score, threshold_keeping

DEPLOY = Paths().root / "deploy"


class Deployable(nn.Module):
    """A plain (merged) ViTForImageClassification that also returns its CLS embedding."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        h = self.model.vit(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.model.classifier(h), h


def export_onnx(model, path: Path, image_size: int = 224) -> None:
    """Export with a dynamic batch dimension; inputs `pixel_values`, outputs `logits` and `features`."""
    wrapper = Deployable(model).eval().cpu()
    dummy = torch.randn(2, 3, image_size, image_size)
    torch.onnx.export(wrapper, (dummy,), str(path), input_names=["pixel_values"],
                      output_names=["logits", "features"],
                      dynamic_shapes={"pixel_values": {0: torch.export.Dim("batch")}}, dynamo=True)


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic int8 quantization of the weights (activations stay float), for CPU inference."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # The exporter records intermediate shapes (value_info) that onnx's shape inference rejects
    # during quantization; dropping them is safe because the quantizer re-infers every shape.
    model = onnx.load(str(src))
    del model.graph.value_info[:]
    clean = src.with_name(src.stem + "_noinfo.onnx")
    onnx.save(model, str(clean))
    quantize_dynamic(str(clean), str(dst), weight_type=QuantType.QInt8)
    clean.unlink()


def _collect(run: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], loader) -> Tuple[np.ndarray, ...]:
    feats, logits, labels = [], [], []
    for px, y in loader:
        lg, f = run(px.numpy())
        feats.append(f)
        logits.append(lg)
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(labels)


def _median_ms(fn: Callable[[], object], n: int = 50) -> float:
    for _ in range(5):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def _size_mb(path: Path) -> float:
    """Model size including any external weights file (the exporter writes large fp32 weights to <name>.data)."""
    data = path.with_name(path.name + ".data")
    return (path.stat().st_size + (data.stat().st_size if data.exists() else 0)) / 1e6


def _pick_examples(rows_and_decisions: List[Tuple[dict, str]], wanted: str) -> dict:
    return next(row for row, status in rows_and_decisions if status == wanted)


def main() -> None:
    from .train import fit_model  # heavy import, only needed when actually training

    paths = Paths()
    out = DEPLOY / "model"
    out.mkdir(parents=True, exist_ok=True)
    work = paths.root / "checkpoints"
    work.mkdir(exist_ok=True)

    fitted = fit_model("lora", seed=SEED)
    model = fitted["model"].merge_and_unload().cpu().eval()
    loaders = {k: fitted[f"{k}_dl"] for k in ("train", "val", "id_test", "ood")}

    fp32, int8 = work / "model_fp32.onnx", out / "model_int8.onnx"
    export_onnx(model, fp32)
    quantize_int8(fp32, int8)

    import onnxruntime as ort

    sessions = {name: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for name, p in (("onnx_fp32", fp32), ("onnx_int8", int8))}
    wrapper = Deployable(model)

    @torch.no_grad()
    def torch_run(px):
        lg, f = wrapper(torch.from_numpy(px))
        return lg.numpy(), f.numpy()

    backends = {"pytorch": torch_run}
    for name, sess in sessions.items():
        backends[name] = lambda px, s=sess: tuple(s.run(["logits", "features"], {"pixel_values": px}))

    results: Dict = {"seed": SEED, "backends": {}}
    data = {}
    for name, run in backends.items():
        data[name] = {split: _collect(run, dl) for split, dl in loaders.items()}
        d = data[name]
        detector = MahalanobisDetector.fit(d["train"][0], d["train"][2])
        results["backends"][name] = evaluate_detectors(detector, d["val"], d["id_test"], d["ood"])
        if name == "onnx_int8":
            deployed = detector

    ref = data["pytorch"]
    for name in sessions:
        for split in ("id_test", "ood"):
            results["backends"][name][f"{split}_agreement_with_pytorch"] = float(
                (data[name][split][1].argmax(1) == ref[split][1].argmax(1)).mean())
            results["backends"][name][f"{split}_max_abs_logit_diff"] = float(
                np.abs(data[name][split][1] - ref[split][1]).max())

    # Ship the detector fit on the int8 model's own features, with thresholds from validation images.
    val_f, val_l, _ = data["onnx_int8"]["val"]
    shift_thr = threshold_keeping(deployed.score(val_f))
    familiar = deployed.score(val_f) <= shift_thr
    conf_thr = float(np.quantile(1 - msp_score(val_l)[familiar], 1 - KEEP_CONFIDENT))
    save_detector(out / "detector.npz", deployed.means, deployed.precision, shift_thr, conf_thr, CLASS_NAMES)

    px1 = next(iter(loaders["id_test"]))[0][:1].numpy()
    results["latency_ms_batch1_cpu"] = {
        "pytorch": _median_ms(lambda: torch_run(px1)),
        **{name: _median_ms(lambda s=sess: s.run(None, {"pixel_values": px1})) for name, sess in sessions.items()},
    }
    results["size_mb"] = {"onnx_fp32": _size_mb(fp32), "onnx_int8": _size_mb(int8)}

    # Two example inputs for the app: a dermoscopy image it answers, and a smartphone photo it refuses.
    clf = SkinClassifier(out)
    pairs = []
    for split in ("id_test", "ood"):
        f, lg, _ = data["onnx_int8"][split]
        for row, fi, li in zip(loaders[split].dataset.rows, f, lg):
            pairs.append((row, decide(li, fi, clf.means, clf.precision, clf.shift_threshold,
                                      clf.confidence_threshold, clf.class_names).status))
    ex_dir = DEPLOY / "examples"
    ex_dir.mkdir(exist_ok=True)
    credits = []
    for row, fname in ((_pick_examples(pairs[:len(loaders["id_test"].dataset)], "prediction"), "dermoscopy_example.jpg"),
                       (_pick_examples(pairs[len(loaders["id_test"].dataset):], "unfamiliar"), "smartphone_example.jpg")):
        shutil.copy(paths.data / row["path"], ex_dir / fname)
        credits.append(f"- `{fname}`: {row['isic_id']} ({DISPLAY_NAMES[row['class_short']]}), "
                       f"{row['attribution']}, {row['copyright_license']}, via the ISIC Archive")
    (ex_dir / "ATTRIBUTION.md").write_text("# Example image credits\n\n" + "\n".join(credits) + "\n", encoding="utf-8")

    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "deploy_benchmark.json").write_text(json.dumps(results, indent=2))
    summary = {k: results[k] for k in ("latency_ms_batch1_cpu", "size_mb")}
    summary["int8"] = {k: v for k, v in results["backends"]["onnx_int8"].items() if k in ("detectors", "ood", "id_test")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
