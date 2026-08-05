"""register_kind/get_kind_function/register_replot/get_replot_function — the check-kind function registries."""

from __future__ import annotations

_KIND_FUNCTIONS: dict = {}
_REPLOT_FUNCTIONS: dict = {}


def register_kind(name: str):
    """Decorator: registers fn under name in _KIND_FUNCTIONS."""
    def deco(fn):
        _KIND_FUNCTIONS[name] = fn
        return fn
    return deco


def get_kind_function(name: str):
    """Raises KeyError if name isn't registered."""
    try:
        return _KIND_FUNCTIONS[name]
    except KeyError:
        raise KeyError(f"unknown check kind {name!r}; known: {sorted(_KIND_FUNCTIONS)}") from None


def register_replot(name: str):
    """Decorator: registers a replot_<kind> fn under name in _REPLOT_FUNCTIONS."""
    def deco(fn):
        _REPLOT_FUNCTIONS[name] = fn
        return fn
    return deco


def get_replot_function(name: str):
    """Raises KeyError if name has no registered replot function."""
    try:
        return _REPLOT_FUNCTIONS[name]
    except KeyError:
        raise KeyError(f"no replot function registered for check kind {name!r}; known: {sorted(_REPLOT_FUNCTIONS)}") from None
