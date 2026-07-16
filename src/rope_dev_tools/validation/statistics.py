"""register_statistic/get_statistic_function registry — statistics are opt-in, never computed by default."""

from __future__ import annotations

import numpy as np

_STATISTIC_FUNCTIONS: dict = {}

_LOG_FLOOR = 1e-300


def register_statistic(name: str):
    def deco(fn):
        _STATISTIC_FUNCTIONS[name] = fn
        return fn
    return deco


def get_statistic_function(name: str):
    try:
        return _STATISTIC_FUNCTIONS[name]
    except KeyError:
        raise KeyError(f"unknown statistic {name!r}; known: {sorted(_STATISTIC_FUNCTIONS)}") from None


def compute_statistics(predicted, truth, names: "list[str] | None") -> "dict[str, float] | None":
    """Returns None if names is None/empty."""
    if not names:
        return None
    return {name: get_statistic_function(name)(predicted, truth) for name in names}


def format_statistics_text(statistics: "dict[str, float] | None") -> "str | None":
    """'log_bias: 1.23e-14\\nlog_rmse: 3.40e-02', or None."""
    if not statistics:
        return None
    return "\n".join(f"{name}: {value:.3g}" for name, value in statistics.items())


@register_statistic("bias")
def bias(predicted, truth) -> float:
    return float(np.mean(np.asarray(predicted) - np.asarray(truth)))


@register_statistic("rmse")
def rmse(predicted, truth) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - np.asarray(truth)))))


def _log10_floored(x) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=float), _LOG_FLOOR, None))


@register_statistic("log_bias")
def log_bias(predicted, truth) -> float:
    return float(np.mean(_log10_floored(predicted) - _log10_floored(truth)))


@register_statistic("log_rmse")
def log_rmse(predicted, truth) -> float:
    return float(np.sqrt(np.mean(np.square(_log10_floored(predicted) - _log10_floored(truth)))))
