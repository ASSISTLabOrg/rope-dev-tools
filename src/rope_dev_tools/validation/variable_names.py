"""resolve_variable_names — maps a fixed set of field names to actual names in a raw data file, override or default."""

from __future__ import annotations


def resolve_variable_names(available: set, variable_names: "dict | None", defaults: dict, *,
                            error_cls, noun: str, available_noun: str) -> dict:
    """Maps each of defaults' fields to its actual name in available, override or default; raises error_cls if missing."""
    variable_names = variable_names or {}
    resolved = {}
    for field, default in defaults.items():
        name = variable_names.get(field, default)
        if name not in available:
            raise error_cls(
                f"could not find a {noun} named {name!r} for {field!r}; "
                f"available {available_noun}: {sorted(available)}. Pass "
                f"variable_names={{{field!r}: 'actual_name'}} to override."
            )
        resolved[field] = name
    return resolved
