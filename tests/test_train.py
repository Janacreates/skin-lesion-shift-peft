import numpy as np

from skinshift.data import CLASS_NAMES
from skinshift.model import load_model_and_processor
from skinshift.train import apply_method, expected_calibration_error


def test_ece_is_zero_when_perfectly_calibrated():
    # 100 predictions at 0.9 confidence, 90 correct -> confidence matches accuracy exactly
    confidence = np.full(100, 0.9)
    correct = np.array([True] * 90 + [False] * 10)
    assert expected_calibration_error(confidence, correct) < 1e-9  # exact 0 up to float error


def test_ece_is_positive_when_overconfident():
    # 100 predictions all at 0.99 confidence, only 50% correct -> badly overconfident
    confidence = np.full(100, 0.99)
    correct = np.array([True] * 50 + [False] * 50)
    ece = expected_calibration_error(confidence, correct)
    assert ece > 0.4


def _param_ids(model):
    return {id(p) for p in model.parameters() if p.requires_grad}


def test_linear_probe_only_trains_the_classifier():
    model, _ = load_model_and_processor(num_labels=len(CLASS_NAMES))
    model, n_trainable, n_total = apply_method(model, "linear_probe")
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    assert n_trainable == classifier_params
    assert n_trainable < n_total


def test_full_finetune_trains_everything():
    model, _ = load_model_and_processor(num_labels=len(CLASS_NAMES))
    model, n_trainable, n_total = apply_method(model, "full")
    assert n_trainable == n_total


def test_lora_trains_far_fewer_params_than_full_but_more_than_linear_probe():
    m1, _ = load_model_and_processor(num_labels=len(CLASS_NAMES))
    _, n_linear, n_total = apply_method(m1, "linear_probe")
    m2, _ = load_model_and_processor(num_labels=len(CLASS_NAMES))
    _, n_lora, _ = apply_method(m2, "lora")
    assert n_linear < n_lora < n_total
