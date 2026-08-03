"""register_kind/get_kind_function — the in-repo check-kind registry, keyed by kind string."""

from __future__ import annotations

_KIND_FUNCTIONS: dict = {}


def register_kind(name: str):
    def deco(fn):
        _KIND_FUNCTIONS[name] = fn
        return fn
    return deco


def get_kind_function(name: str):
    try:
        return _KIND_FUNCTIONS[name]
    except KeyError:
        raise KeyError(f"unknown check kind {name!r}; known: {sorted(_KIND_FUNCTIONS)}") from None


def passes_threshold(value: float, threshold: dict) -> bool:
    if "max" in threshold and value > threshold["max"]:
        return False
    if "min" in threshold and value < threshold["min"]:
        return False
    return True


def delta_label(base_label: str, delta_hours, *, n_deltas: int) -> str:
    """base_label, suffixed with the start_delta in hours -- but only when there's more than one
    delta to disambiguate (a period with a single start_deltas entry, [0] or otherwise, keeps its
    exact unsuffixed label, matching pre-start_deltas behavior)."""
    if n_deltas <= 1:
        return base_label
    return f"{base_label} (Δ{delta_hours:+d}h)"


def delta_suffix(delta_hours, *, n_deltas: int) -> str:
    """Filename-safe suffix for the same disambiguation, e.g. '_delta-48'/'_delta+24' -- empty
    when there's nothing to disambiguate."""
    if n_deltas <= 1:
        return ""
    return f"_delta{delta_hours:+d}"


def delta_stat_key(delta_hours) -> str:
    """Statistics/output JSON nesting key for one start_delta -- always present regardless of
    n_deltas, so the output shape doesn't silently change based on suite content."""
    return f"delta_{delta_hours:+d}h"
