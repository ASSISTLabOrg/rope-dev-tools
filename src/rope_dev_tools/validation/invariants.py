"""Forecast invariants that must hold regardless of what a validation suite's
checks specify, per .claude/rules/forecast-invariants.md: uncertainty always
accompanies density, both are non-negative, and inference is deterministic.
Runs unconditionally in both wrapper mode and exported-dir mode.
"""

from __future__ import annotations


class InvariantViolation(AssertionError):
    pass


def assert_forecast_invariants(model, probe_points) -> None:
    """probe_points: iterable of (time, lst, lat, alt_km) tuples to query.

    Each point is queried twice (to check determinism) and validated for
    non-negativity and uncertainty presence. Raises immediately on the first
    violation — this is a fail-loud assertion, not a Result the caller can
    choose to ignore.
    """
    for time, lst, lat, alt_km in probe_points:
        first = model.query(time, lst, lat, alt_km)
        second = model.query(time, lst, lat, alt_km)

        for label, result in (("first", first), ("second", second)):
            if "density" not in result or "uncertainty" not in result:
                raise InvariantViolation(
                    f"query({time}, {lst}, {lat}, {alt_km}) {label} call is missing "
                    f"'density' or 'uncertainty': {result!r}"
                )
            if result["density"] < 0:
                raise InvariantViolation(
                    f"query({time}, {lst}, {lat}, {alt_km}) {label} call returned "
                    f"negative density: {result['density']}"
                )
            if result["uncertainty"] < 0:
                raise InvariantViolation(
                    f"query({time}, {lst}, {lat}, {alt_km}) {label} call returned "
                    f"negative uncertainty: {result['uncertainty']}"
                )

        if first != second:
            raise InvariantViolation(
                f"query({time}, {lst}, {lat}, {alt_km}) is not deterministic: "
                f"{first!r} != {second!r}"
            )
