import numpy as np
import torch

from skinshift.attention import central_mask, central_mass, gradient_rollout, spread


def _one_layer(cls_row, grad_sign=1.0, patches=4):
    """One layer, one head: the CLS token attends with `cls_row`; patches attend only to themselves."""
    t = patches + 1
    a = torch.eye(t)
    a[0] = torch.tensor(cls_row, dtype=torch.float32)
    a = a[None, None]                        # (batch, heads, tokens, tokens)
    return [a], [torch.full_like(a, grad_sign)]


def test_rollout_puts_relevance_where_the_cls_token_attends():
    att, grads = _one_layer([0.1, 0.0, 0.0, 0.9, 0.0])
    m = gradient_rollout(att, grads)[0].numpy()
    assert m.shape == (4,) and np.isclose(m.sum(), 1.0)
    assert m.argmax() == 2                    # patch index 2 = token 3


def test_negative_gradients_count_as_no_evidence():
    att, grads = _one_layer([0.1, 0.0, 0.0, 0.9, 0.0], grad_sign=-1.0)
    m = gradient_rollout(att, grads)[0].numpy()
    np.testing.assert_allclose(m, 0.25)       # nothing positive anywhere -> uniform, not NaN


def test_central_region_is_eight_by_eight_and_uniform_maps_score_its_area():
    mask = central_mask()
    assert mask.sum() == 64 and mask.size == 196
    uniform = np.full((1, 196), 1 / 196)
    assert np.isclose(central_mass(uniform)[0], 64 / 196)


def test_spread_is_one_for_uniform_and_zero_for_a_single_patch():
    uniform = np.full((1, 196), 1 / 196)
    peaked = np.zeros((1, 196))
    peaked[0, 50] = 1.0
    assert np.isclose(spread(uniform)[0], 1.0)
    assert np.isclose(spread(peaked)[0], 0.0, atol=1e-6)
