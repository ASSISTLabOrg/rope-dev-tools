"""register_kind/get_kind_function registry, dispatched by rope-registry's check_kinds.json kind strings."""

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
