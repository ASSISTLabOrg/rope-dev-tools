"""Check-kind registry (see base.py) plus shared per-check formatting helpers (threshold/start-delta)."""

from __future__ import annotations

from rope_dev_tools.validation.checks.base import (
    get_kind_function,
    get_replot_function,
    register_kind,
    register_replot,
)

__all__ = [
    "register_kind", "get_kind_function",
    "register_replot", "get_replot_function",
    "passes_threshold", "delta_label", "delta_suffix", "delta_stat_key",
]


def passes_threshold(value: float, threshold: dict) -> bool:
    """threshold may have "min" and/or "max"; both bounds inclusive."""
    if "max" in threshold and value > threshold["max"]:
        return False
    if "min" in threshold and value < threshold["min"]:
        return False
    return True


def delta_label(base_label: str, delta_hours, *, n_deltas: int) -> str:
    """base_label with a '(Δ{delta}h)' suffix, or unsuffixed when n_deltas <= 1."""
    if n_deltas <= 1:
        return base_label
    return f"{base_label} (Δ{delta_hours:+d}h)"


def delta_suffix(delta_hours, *, n_deltas: int) -> str:
    """Filename-safe '_delta{delta}' suffix, or '' when n_deltas <= 1."""
    if n_deltas <= 1:
        return ""
    return f"_delta{delta_hours:+d}"


def delta_stat_key(delta_hours) -> str:
    """Statistics/output JSON nesting key for one start_delta -- always present regardless of n_deltas."""
    return f"delta_{delta_hours:+d}h"
