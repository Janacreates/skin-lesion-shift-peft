"""Model loading: a small pretrained ViT, ready for a linear probe, full fine-tune, or LoRA.

Checkpoint choice (Day 1): WinKawaks/vit-tiny-patch16-224 (~5.5M params) — small enough to
iterate fast on a laptop in a one-week budget, and a standard transformers ViT, so PEFT's LoRA
config (which targets nn.Linear attention-projection layers) applies with no custom code.
"""
from __future__ import annotations

from typing import Tuple

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

CHECKPOINT = "WinKawaks/vit-tiny-patch16-224"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_processor(num_labels: int, checkpoint: str = CHECKPOINT
                              ) -> Tuple[ViTForImageClassification, ViTImageProcessor]:
    processor = ViTImageProcessor.from_pretrained(checkpoint)
    model = ViTForImageClassification.from_pretrained(
        checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True)
    return model, processor
