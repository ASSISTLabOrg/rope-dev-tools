"""Statistics registry: built-ins, opt-in dispatch, and text formatting."""

from __future__ import annotations

import numpy as np
import pytest

from rope_dev_tools.validation.statistics import (
    compute_statistic_uncertainties,
    compute_statistics,
    format_statistics_text,
    get_statistic_function,
)


def test_all_statistics():
    predicted, truth = np.array([1.0e-10, 2.0e-10, 3.0e-10]), np.array([1.0e-10, 2.0e-10, 4.0e-10])
    # only the 3rd pair has a nonzero raw ln-diff -- the other two are identical (diff 0).
    diffs = np.array([0.0, 0.0, np.log(3) - np.log(4)])
    _ratio = np.exp(np.mean(diffs))
    _bias = (_ratio - 1.0) * 100.0
    _rmse = np.sqrt(np.mean(diffs**2)) * 100.0
    # std subtracts ln(ratio) from *every* point (not just the nonzero one) before squaring, so
    # the two zero-diff points contribute too once de-biased.
    _std = np.sqrt(np.mean((diffs - np.log(_ratio))**2)) * 100.0

    assert compute_statistics(predicted, truth, names=["bias"])["bias"] == pytest.approx(_bias)
    assert compute_statistics(predicted, truth, names=["rmse"])["rmse"] == pytest.approx(_rmse)
    assert compute_statistics(predicted, truth, names=["std"])["std"] == pytest.approx(_std)


def test_unsigned_bias_averages_the_error_magnitude_before_exponentiating():
    # unsigned_bias = (exp(mean(|ln(predicted) - ln(truth)|)) - 1) * 100 -- NOT abs(bias()), since
    # averaging |d_i| before exp() differs from exp(|mean(d_i)|) whenever errors partially cancel.
    predicted, truth = np.array([1.0e-10, 2.0e-10, 3.0e-10]), np.array([1.0e-10, 2.0e-10, 4.0e-10])
    diffs = np.array([0.0, 0.0, np.log(3) - np.log(4)])
    expected = (np.exp(np.mean(np.abs(diffs))) - 1.0) * 100.0

    signed = compute_statistics(predicted, truth, names=["bias"])["bias"]
    unsigned = compute_statistics(predicted, truth, names=["unsigned_bias"])["unsigned_bias"]
    assert signed < 0.0  # predicted is low relative to truth here, so signed bias is negative...
    assert unsigned == pytest.approx(expected)
    assert unsigned >= abs(signed)  # Jensen's inequality: mean(|d_i|) >= |mean(d_i)|, always


def test_std_is_order_independent_of_bias_in_the_requested_names():
    # std computes its own bias internally now, so it must not care whether "bias" was also
    # requested, or in what order -- this used to crash with KeyError when std came first.
    predicted, truth = np.array([1.0e-10, 2.0e-10, 3.0e-10]), np.array([1.0e-10, 2.0e-10, 4.0e-10])
    std_only = compute_statistics(predicted, truth, names=["std"])["std"]
    std_first = compute_statistics(predicted, truth, names=["std", "bias"])["std"]
    bias_first = compute_statistics(predicted, truth, names=["bias", "std"])["std"]
    assert std_only == pytest.approx(std_first) == pytest.approx(bias_first)


def test_unknown_statistic_raises():
    with pytest.raises(KeyError):
        get_statistic_function("nonexistent_stat")


def test_compute_statistics_returns_none_when_no_names_requested():
    predicted, truth = np.array([1.0]), np.array([1.0])
    assert compute_statistics(predicted, truth, None) is None
    assert compute_statistics(predicted, truth, []) is None


def test_compute_statistics_computes_only_requested_names():
    predicted, truth = np.array([1.0, 2.0]), np.array([1.0, 3.0])
    result = compute_statistics(predicted, truth, ["bias"])
    assert set(result) == {"bias"}


def test_format_statistics_text():
    assert format_statistics_text(None) is None
    text = format_statistics_text({"bias": 1.23456e-14})
    assert "bias" in text and "1.23e-14" in text


def test_format_statistics_text_with_uncertainties():
    text = format_statistics_text({"bias": 3.34, "rmse": 9.01}, {"bias": 1.2})
    lines = text.split("\n")
    assert lines[0] == "bias: 3.34 ± 1.2 %"
    assert lines[1] == "rmse: 9.01 %"  # no entry in uncertainties -- plain format, unchanged


def test_compute_statistic_uncertainties_returns_none_without_names_or_uncert():
    predicted, truth = np.array([1.0, 2.0]), np.array([1.0, 3.0])
    assert compute_statistic_uncertainties(predicted, truth, np.array([0.1, 0.1]), None) is None
    assert compute_statistic_uncertainties(predicted, truth, None, ["bias"]) is None


def test_compute_statistic_uncertainties_silently_omits_unregistered_names():
    predicted, truth = np.array([1.0e-10, 2.0e-10]), np.array([1.0e-10, 2.2e-10])
    predicted_uncert = predicted * 0.05
    result = compute_statistic_uncertainties(predicted, truth, predicted_uncert, ["bias", "unsigned_bias"])
    assert set(result) == {"bias"}  # unsigned_bias has no registered uncertainty function


def test_compute_statistic_uncertainties_matches_linearized_monte_carlo():
    # predicted_uncert is treated as independent across points -- perturbations are drawn per-point.
    rng = np.random.default_rng(0)
    truth = rng.uniform(1e-13, 1e-9, 12)
    predicted = truth * rng.uniform(0.7, 1.4, 12)
    predicted_uncert = predicted * rng.uniform(0.01, 0.1, 12)

    analytic = compute_statistic_uncertainties(predicted, truth, predicted_uncert, ["bias", "rmse", "std"])

    from rope_dev_tools.validation.statistics import bias, rmse, std
    samples = {"bias": [], "rmse": [], "std": []}
    for _ in range(20000):
        p = predicted + rng.normal(0, predicted_uncert)
        samples["bias"].append(bias(p, truth))
        samples["rmse"].append(rmse(p, truth))
        samples["std"].append(std(p, truth))

    for name in ("bias", "rmse", "std"):
        mc_sigma = np.std(samples[name])
        assert analytic[name] == pytest.approx(mc_sigma, rel=0.15)  # linearization slack, not exact


def test_compute_statistic_uncertainties_raises_clearly_when_rmse_or_std_is_exactly_zero():
    # predicted == truth everywhere -> rmse == std == 0, a genuine singularity of the linearized
    # formula (sqrt has no derivative at 0), not just a numerically-large result.
    predicted = truth = np.array([1.0e-10, 2.0e-10, 3.0e-10])
    predicted_uncert = predicted * 0.05
    with pytest.raises(ZeroDivisionError, match="rmse"):
        compute_statistic_uncertainties(predicted, truth, predicted_uncert, ["rmse"])
    with pytest.raises(ZeroDivisionError, match="std"):
        compute_statistic_uncertainties(predicted, truth, predicted_uncert, ["std"])


def test_compute_statistic_uncertainties_is_shape_agnostic():
    # lonlat_snapshot_series stacks multiple (n_lst, n_lat) grids into one 3D array -- the N in the
    # 1/N propagation prefactor must count every element, not just the first axis.
    rng = np.random.default_rng(1)
    truth_flat = rng.uniform(1e-13, 1e-9, 24)
    predicted_flat = truth_flat * rng.uniform(0.8, 1.2, 24)
    uncert_flat = predicted_flat * 0.05

    flat = compute_statistic_uncertainties(predicted_flat, truth_flat, uncert_flat, ["bias", "rmse", "std"])
    nested = compute_statistic_uncertainties(
        predicted_flat.reshape(4, 2, 3), truth_flat.reshape(4, 2, 3), uncert_flat.reshape(4, 2, 3),
        ["bias", "rmse", "std"],
    )
    for name in ("bias", "rmse", "std"):
        assert nested[name] == pytest.approx(flat[name])
