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

def format_statistics_text(statistics: "dict[str, float] | None") -> "str | None":
    """'log_bias: 1.23e-14\\nlog_rmse: 3.40e-02', or None."""
    if not statistics:
        return None
    return "\n".join(f"{name}: {value:.3g}" for name, value in statistics.items())

def _ln_floored(x) -> np.ndarray:
    return np.log(np.clip(np.asarray(x, dtype=float), _LOG_FLOOR, None))

@register_statistic("bias")
def bias(predicted, truth) -> float:
    return float(np.exp(np.mean(_ln_floored(predicted) - _ln_floored(truth))))

@register_statistic("rmse")
def rmse(predicted, truth) -> float:
    return float(np.sqrt(np.mean((_ln_floored(predicted) - _ln_floored(truth))**2)))

@register_statistic("std")
def std(predicted, truth, bias=np.exp(_LOG_FLOOR)) -> float:
    return float(np.sqrt(np.mean(((_ln_floored(predicted) - _ln_floored(truth)) - _ln_floored(bias))**2)))

def compute_statistics(predicted, truth, names: "list[str] | None") -> "dict[str, float] | None":
    """Returns None if names is None/empty."""
    if not names:
        return None

    vals = {}
    for name in names:
        match name:
            case "bias":
                vals["bias"] = bias(predicted, truth)
            case "rmse":
                vals["rmse"] = rmse(predicted, truth)
            case "std":
                if "bias" not in names:
                    _bias = bias(predicted, truth)
                    vals["std"] = std(predicted, truth, _bias)
                else:
                    vals["std"] = std(predicted, truth, vals["bias"])
            case _:
                raise KeyError(f"unknown statistic {name!r}; known: {sorted(_STATISTIC_FUNCTIONS)}") from None
    
    return vals