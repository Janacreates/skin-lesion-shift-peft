import numpy as np

from skinshift.report import _mean_std, comparison_table, risk_coverage, welch_t_test


def test_risk_coverage_returns_decreasing_coverage_as_threshold_rises():
    # regression test for a real bug: risk_coverage()'s coverage array comes back in
    # DEcreasing order (thresholds increase through the loop), and feeding that directly into
    # np.interp (which requires an ascending x-array) silently produced a near-flat, wrong
    # curve. This locks in the documented direction so a fix can't quietly flip it back.
    rng = np.random.default_rng(0)
    n = 500
    probs = rng.dirichlet(np.ones(4), size=n)
    labels = rng.integers(0, 4, size=n)
    cov, acc = risk_coverage(probs, labels)
    assert cov[0] > cov[-1]  # coverage=1 (keep everyone) is the FIRST point (threshold~0)
    assert np.all(np.diff(cov) <= 1e-9)  # non-increasing throughout


def test_reversed_risk_coverage_interpolates_correctly():
    # the exact fix: reversing before np.interp must round-trip through the original points
    rng = np.random.default_rng(1)
    n = 500
    probs = rng.dirichlet(np.ones(4), size=n)
    labels = rng.integers(0, 4, size=n)
    cov, acc = risk_coverage(probs, labels, n_points=15)

    grid = 100 * cov[::-1]  # interpolate exactly at the known points -> should reproduce them
    interpolated = np.interp(grid, 100 * cov[::-1], 100 * acc[::-1])
    np.testing.assert_allclose(interpolated, 100 * acc[::-1], atol=1e-6)

    # the bug this guards against: feeding the UN-reversed (decreasing) array into np.interp
    # gives something different from the correct, reversed-array result whenever the curve
    # actually varies -- if this ever stops failing, the two are being computed the same way
    # and the assertion above stops proving anything.
    wrong = np.interp(grid, 100 * cov, 100 * acc)
    if np.ptp(acc) > 1e-9:
        assert not np.allclose(wrong, interpolated)


def test_mean_std_matches_numpy_sample_std():
    vals = [0.60, 0.62, 0.58, 0.65, 0.61]
    out = _mean_std(vals)
    assert out["mean"] == np.mean(vals)
    assert out["std"] == np.std(vals, ddof=1)


def test_mean_std_zero_std_for_a_single_value():
    out = _mean_std([0.7])
    assert out["mean"] == 0.7
    assert out["std"] == 0.0  # ddof=1 std of one point is undefined; we report 0, not NaN


def test_welch_t_test_zero_diff_for_identical_groups():
    a = [0.5, 0.52, 0.48, 0.51, 0.49]
    out = welch_t_test(a, a)
    assert abs(out["diff"]) < 1e-9
    assert out["t"] == 0.0


def test_welch_t_test_sign_matches_direction_of_difference():
    higher = [0.70, 0.72, 0.68, 0.71, 0.69]
    lower = [0.50, 0.52, 0.48, 0.51, 0.49]
    out = welch_t_test(higher, lower)
    assert out["diff"] > 0 and out["t"] > 0
    out_rev = welch_t_test(lower, higher)
    assert out_rev["diff"] < 0 and out_rev["t"] < 0


def _fake_run(method, seed, id_acc, ood_acc):
    return {
        "method": method, "seed": seed, "trainable_pct": 1.0, "train_seconds": 10.0,
        "id_test": {"accuracy": id_acc, "macro_f1": id_acc, "ece": 0.1},
        "ood_test": {"accuracy": ood_acc, "macro_f1": ood_acc, "ece": 0.2},
    }


def test_comparison_table_aggregates_across_seeds():
    runs = {"lora": [_fake_run("lora", s, 0.7, 0.5 + 0.01 * s) for s in range(5)]}
    table = comparison_table(runs)
    row = table[0]
    assert row["method"] == "lora"
    assert row["n_seeds"] == 5
    assert row["id_accuracy_mean"] == 0.7
    assert row["id_accuracy_std"] == 0.0  # identical id accuracy every seed here
    assert row["ood_accuracy_std"] > 0  # ood accuracy varies by seed here
