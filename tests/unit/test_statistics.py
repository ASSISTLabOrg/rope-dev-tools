"""Statistics registry: built-ins, opt-in dispatch, and text formatting."""

from __future__ import annotations

import numpy as np
import pytest

from rope_dev_tools.validation.statistics import (
    compute_statistics,
    format_statistics_text,
)


def test_all_statistics():
    predicted, truth = np.array([1.0e-10, 2.0e-10, 3.0e-10]), np.array([1.0e-10, 2.0e-10, 4.0e-10])
    _bias = np.exp((np.log(3) - np.log(4)) / 3)
    _rmse = np.sqrt((np.log(3) - np.log(4))**2 / 3)
    _std = np.sqrt(((np.log(3) - np.log(4)) - np.log(_bias))**2 / 3)
    assert compute_statistics(predicted, truth, names=["bias"]) == pytest.approx(_bias)
    assert compute_statistics(predicted, truth, names=["rmse"]) == pytest.approx(_rmse)
    assert compute_statistics(predicted, truth, names=["std"]) == pytest.approx(_std)

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
