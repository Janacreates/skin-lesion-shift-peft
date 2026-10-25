import numpy as np
import torch

from deploy.core import mahalanobis_min
from skinshift.data import CLASS_NAMES
from skinshift.model import load_model_and_processor
from skinshift.shift import (MahalanobisDetector, embed, energy_score, evaluate_detectors, msp_score,
                             threshold_keeping)


def _clusters(rng, n_per_class=200, dim=8, shift=0.0):
    """Four Gaussian classes in `dim` dimensions, optionally all moved by `shift`."""
    means = np.eye(4, dim) * 6.0
    feats = np.concatenate([rng.normal(m + shift, 1.0, size=(n_per_class, dim)) for m in means])
    labels = np.repeat(np.arange(4), n_per_class)
    return feats, labels


def test_mahalanobis_is_near_zero_at_class_means_and_grows_with_distance():
    rng = np.random.default_rng(0)
    feats, labels = _clusters(rng)
    det = MahalanobisDetector.fit(feats, labels)
    at_means = det.score(det.means)
    far = det.score(det.means + 10.0)
    assert np.all(at_means < 1e-6)
    assert np.all(far > 50)


def test_mahalanobis_with_identity_precision_is_squared_distance_to_nearest_mean():
    means = np.array([[0.0, 0.0], [10.0, 0.0]])
    det = MahalanobisDetector(means=means, precision=np.eye(2))
    x = np.array([[1.0, 2.0], [9.0, 0.0]])
    np.testing.assert_allclose(det.score(x), [5.0, 1.0])
    # the app's numpy-only copy must give the same numbers
    np.testing.assert_allclose(mahalanobis_min(x, means, np.eye(2)), det.score(x))


def test_baseline_scores_rank_uncertain_logits_as_more_unusual():
    logits = np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert msp_score(logits)[1] > msp_score(logits)[0]
    assert energy_score(logits)[1] > energy_score(logits)[0]


def test_threshold_keeping_lets_the_requested_fraction_pass():
    scores = np.arange(1000, dtype=float)
    thr = threshold_keeping(scores, keep=0.95)
    assert abs(np.mean(scores <= thr) - 0.95) < 0.002


def test_detectors_separate_a_clearly_shifted_set():
    rng = np.random.default_rng(1)
    train = _clusters(rng)
    val, id_test = _clusters(rng, n_per_class=50), _clusters(rng, n_per_class=50)
    ood = _clusters(rng, n_per_class=50, shift=8.0)
    det = MahalanobisDetector.fit(*train)
    # logits from distance to each class mean, so the baselines have something sensible to score
    to_split = lambda f, y: (f, -((f[:, None, :] - det.means[None]) ** 2).sum(-1), y)  # noqa: E731
    out = evaluate_detectors(det, to_split(*val), to_split(*id_test), to_split(*ood))
    m = out["detectors"]["mahalanobis"]
    assert m["auroc"] > 0.99 and m["ood_caught"] > 0.95 and m["id_test_false_alarm"] < 0.2
    assert out["ood"]["refused_unfamiliar"] > 0.95
    assert 0.0 <= out["id_test"]["answered"] <= 1.0


def test_embed_returns_the_features_the_classifier_actually_sees():
    model, _ = load_model_and_processor(num_labels=len(CLASS_NAMES))
    torch.manual_seed(0)
    px = torch.randn(3, 3, 224, 224)
    loader = [(px, torch.tensor([0, 1, 2]))]
    feats, logits, labels = embed(model, loader, torch.device("cpu"))
    with torch.no_grad():
        expected = model(pixel_values=px).logits.numpy()
    assert feats.shape == (3, model.config.hidden_size)
    np.testing.assert_allclose(logits, expected, atol=1e-5)
    assert labels.tolist() == [0, 1, 2]
