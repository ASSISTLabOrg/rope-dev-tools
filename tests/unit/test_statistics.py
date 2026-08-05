"""Statistics registry: built-ins, opt-in dispatch, and text formatting."""

from __future__ import annotations

import numpy as np
import pytest

from rope_dev_tools.validation.statistics import (
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


def test_unsigned_bias_is_absolute_value_of_bias():
    predicted, truth = np.array([1.0e-10, 2.0e-10, 3.0e-10]), np.array([1.0e-10, 2.0e-10, 4.0e-10])
    signed = compute_statistics(predicted, truth, names=["bias"])["bias"]
    unsigned = compute_statistics(predicted, truth, names=["unsigned_bias"])["unsigned_bias"]
    assert signed < 0.0  # predicted is low relative to truth here, so signed bias is negative...
    assert unsigned == pytest.approx(abs(signed))  # ...and unsigned_bias drops the sign.


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
