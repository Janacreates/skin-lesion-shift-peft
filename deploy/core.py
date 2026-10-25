"""Inference for the demo app, using only numpy, Pillow and ONNX Runtime (no PyTorch in the container).

Every upload goes through two gates before the app answers:
1. Shift gate: if the image's CLS embedding is far from every training class (squared Mahalanobis
   distance above a threshold set so 95% of in-distribution validation images pass), refuse.
2. Confidence gate: if the top probability is below a threshold set on in-distribution validation
   images, say "not confident" instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
MEAN = STD = 0.5  # WinKawaks/vit-tiny-patch16-224's processor: rescale to [0, 1], then (x - 0.5) / 0.5

DISPLAY_NAMES = {
    "melanoma": "Melanoma",
    "bcc": "Basal cell carcinoma",
    "actinic_keratosis": "Actinic keratosis",
    "nevus": "Nevus (mole)",
}


def preprocess(img: Image.Image) -> np.ndarray:
    """Same steps as the model's Hugging Face processor: RGB, bilinear resize to 224x224, normalise."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # (1, 3, 224, 224)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def mahalanobis_min(features: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance from each row of `features` to the nearest class mean."""
    diff = features[:, None, :] - means[None, :, :]
    return np.einsum("ncd,de,nce->nc", diff, precision, diff).min(axis=1)


@dataclass
class Decision:
    status: str  # "unfamiliar", "not_confident" or "prediction"
    shift_score: float
    shift_threshold: float
    confidence: float
    confidence_threshold: float
    probabilities: Dict[str, float] = field(default_factory=dict)

    @property
    def top_class(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)


def decide(logits: np.ndarray, features: np.ndarray, means: np.ndarray, precision: np.ndarray,
           shift_threshold: float, confidence_threshold: float, class_names: Sequence[str]) -> Decision:
    """Apply the shift gate, then the confidence gate, to one image's logits and CLS embedding."""
    shift = float(mahalanobis_min(features.reshape(1, -1), means, precision)[0])
    probs = softmax(logits.reshape(-1))
    conf = float(probs.max())
    probabilities = {name: float(p) for name, p in zip(class_names, probs)}
    if shift > shift_threshold:
        status = "unfamiliar"
    elif conf < confidence_threshold:
        status = "not_confident"
    else:
        status = "prediction"
    return Decision(status, shift, shift_threshold, conf, confidence_threshold, probabilities)


def save_detector(path: Path, means: np.ndarray, precision: np.ndarray, shift_threshold: float,
                  confidence_threshold: float, class_names: Sequence[str]) -> None:
    np.savez(path, means=means.astype(np.float32), precision=precision.astype(np.float32),
             shift_threshold=np.float32(shift_threshold), confidence_threshold=np.float32(confidence_threshold),
             class_names=np.array(class_names))


class SkinClassifier:
    """The int8 ONNX model plus the fitted shift detector, as shipped in deploy/model/."""

    def __init__(self, model_dir: Path, model_file: str = "model_int8.onnx"):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_dir / model_file), providers=["CPUExecutionProvider"])
        d = np.load(model_dir / "detector.npz", allow_pickle=False)
        self.means, self.precision = d["means"], d["precision"]
        self.shift_threshold = float(d["shift_threshold"])
        self.confidence_threshold = float(d["confidence_threshold"])
        self.class_names = [str(c) for c in d["class_names"]]

    def run(self, pixel_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits, features = self.session.run(["logits", "features"], {"pixel_values": pixel_values})
        return logits, features

    def predict(self, img: Image.Image) -> Decision:
        logits, features = self.run(preprocess(img))
        return decide(logits[0], features[0], self.means, self.precision, self.shift_threshold,
                      self.confidence_threshold, self.class_names)
