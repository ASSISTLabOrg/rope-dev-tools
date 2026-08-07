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


def compute_statistic_uncertainties(predicted, truth, predicted_uncert, names: "list[str] | None") -> "dict[str, float] | None":
    """Linearized 1-sigma uncertainty (from predicted_uncert alone) for each of names that has one; silently omits the rest.
    Assumes predicted_uncert is independent across points -- known-incorrect simplification, kept for now."""
    if not names or predicted_uncert is None:
        return None
    bias_ratio = _bias_ratio(predicted, truth)
    by_name = {
        "bias": lambda: _bias_uncert(predicted, predicted_uncert, bias_ratio),
        "rmse": lambda: _rmse_uncert(predicted, truth, predicted_uncert, rmse(predicted, truth)),
        "std": lambda: _std_uncert(predicted, truth, predicted_uncert, std(predicted, truth), bias_ratio),
    }
    return {name: by_name[name]() for name in names if name in by_name}


def format_statistics_text(statistics: "dict[str, float] | None", uncertainties: "dict[str, float] | None" = None) -> "str | None":
    """'bias: 3.34 %\\nrmse: 9.01 ± 1.20 %', or None. uncertainties: optional sibling dict from compute_statistic_uncertainties."""
    if not statistics:
        return None
    uncertainties = uncertainties or {}
    lines = []
    for name, value in statistics.items():
        if name in uncertainties:
            lines.append(f"{name}: {value:.3g} ± {uncertainties[name]:.3g} %")
        else:
            lines.append(f"{name}: {value:.3g} %")
    return "\n".join(lines)


def uncertainty_of_mean(uncert) -> float:
    """Uncertainty of the arithmetic mean, treating per-point uncert as fully correlated: mean(uncert)."""
    return float(np.mean(np.asarray(uncert, dtype=float)))


def _ln_floored(x) -> np.ndarray:
    """ln(x), clipped to _LOG_FLOOR to avoid log(0)."""
    return np.log(np.clip(np.asarray(x, dtype=float), _LOG_FLOOR, None))


def _bias_ratio(predicted, truth) -> float:
    """Raw multiplicative ratio (1.0 = unbiased) -- exp(mean(ln(predicted) - ln(truth)))."""
    return float(np.exp(np.mean(_ln_floored(predicted) - _ln_floored(truth))))


def _unsigned_bias_ratio(predicted, truth) -> float:
    """Raw absolute multiplicative ratio (1.0 = unbiased) -- exp(mean(ln(predicted) - ln(truth)))."""
    return float(np.exp(np.mean(np.abs(_ln_floored(predicted) - _ln_floored(truth)))))


def _bias_uncert(predicted, predicted_uncert, bias_ratio) -> float:
    """Linearized 1-sigma uncertainty on bias%, assuming predicted_uncert is independent across points."""
    n = np.size(predicted)
    return (bias_ratio / n) * np.sqrt(np.sum(predicted_uncert**2 / predicted**2)) * 100.0


def _rmse_uncert(predicted, truth, predicted_uncert, rmse) -> float:
    """Linearized 1-sigma uncertainty on rmse%, assuming predicted_uncert is independent across points."""
    if rmse == 0.0:
        raise ZeroDivisionError(
            "rmse_uncert: rmse is exactly 0 (predicted matches truth exactly everywhere) -- "
            "linearized propagation is singular at rmse=0 (sqrt has no derivative there), not just numerically large"
        )
    n = np.size(predicted)
    d = (_ln_floored(predicted) - _ln_floored(truth)) / predicted
    return 1 / (n * rmse) * np.sqrt(np.sum(d**2 * predicted_uncert**2)) * 100.0**2


def _std_uncert(predicted, truth, predicted_uncert, std, bias_ratio) -> float:
    """Linearized 1-sigma uncertainty on std%, assuming predicted_uncert is independent across points."""
    if std == 0.0:
        raise ZeroDivisionError(
            "std_uncert: std is exactly 0 (residuals against truth are all identical) -- "
            "linearized propagation is singular at std=0 (sqrt has no derivative there), not just numerically large"
        )
    n = np.size(predicted)
    t1 = (_ln_floored(predicted) - _ln_floored(truth)) / predicted
    t2 = np.log(bias_ratio) / predicted
    return 1 / (n * std) * np.sqrt(np.sum((t1 - t2)**2 * predicted_uncert**2)) * 100.0**2


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
    return (_unsigned_bias_ratio(predicted, truth) - 1.0) * 100.0
