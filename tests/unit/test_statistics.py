"""Statistics registry: built-ins, opt-in dispatch, and text formatting."""

from __future__ import annotations

import numpy as np
import pytest

from rope_dev_tools.validation.statistics import (
    compute_statistics,
    format_statistics_text,
    get_statistic_function,
)


def test_bias_and_rmse():
    predicted, truth = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])
    assert get_statistic_function("bias")(predicted, truth) == pytest.approx(-1.0 / 3.0)
    assert get_statistic_function("rmse")(predicted, truth) == pytest.approx(np.sqrt(1.0 / 3.0))


def test_log_bias_and_log_rmse():
    predicted, truth = np.array([1e-12, 1e-12]), np.array([1e-13, 1e-13])
    assert get_statistic_function("log_bias")(predicted, truth) == pytest.approx(1.0, abs=1e-9)
    assert get_statistic_function("log_rmse")(predicted, truth) == pytest.approx(1.0, abs=1e-9)


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
