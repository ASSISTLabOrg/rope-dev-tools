"""register_statistic/get_statistic_function registry — statistics are opt-in, never computed by default."""

from __future__ import annotations

import numpy as np

_STATISTIC_FUNCTIONS: dict = {}

_LOG_FLOOR = 1e-300


def register_statistic(name: str):
    """Decorator: registers fn under name in _STATISTIC_FUNCTIONS."""
    def deco(fn):
        _STATISTIC_FUNCTIONS[name] = fn
        return fn
    return deco


def get_statistic_function(name: str):
    """Raises KeyError if name isn't registered."""
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
    """'bias: 3.34\\nrmse: 9.01', or None."""
    if not statistics:
        return None
    return "\n".join(f"{name}: {value:.3g}" for name, value in statistics.items())


def _ln_floored(x) -> np.ndarray:
    """ln(x), clipped to _LOG_FLOOR to avoid log(0)."""
    return np.log(np.clip(np.asarray(x, dtype=float), _LOG_FLOOR, None))


def _bias_ratio(predicted, truth) -> float:
    """Raw multiplicative ratio (1.0 = unbiased) -- exp(mean(ln(predicted) - ln(truth)))."""
    return float(np.exp(np.mean(_ln_floored(predicted) - _ln_floored(truth))))


@register_statistic("bias")
def bias(predicted, truth) -> float:
    """Percent bias -- (ratio - 1) * 100, 0% is unbiased."""
    return (_bias_ratio(predicted, truth) - 1.0) * 100.0


@register_statistic("rmse")
def rmse(predicted, truth) -> float:
    """Percent RMS of the ln-space error (small-error approximation)."""
    return float(np.sqrt(np.mean((_ln_floored(predicted) - _ln_floored(truth))**2))) * 100.0


@register_statistic("std")
def std(predicted, truth) -> float:
    """Percent standard deviation of the ln-space error around its own bias."""
    ratio = _bias_ratio(predicted, truth)
    return float(np.sqrt(np.mean(((_ln_floored(predicted) - _ln_floored(truth)) - _ln_floored(ratio))**2))) * 100.0


@register_statistic("unsigned_bias")
def unsigned_bias(predicted, truth) -> float:
    """abs(bias()) -- magnitude only, sign dropped."""
    return abs(bias(predicted, truth))
