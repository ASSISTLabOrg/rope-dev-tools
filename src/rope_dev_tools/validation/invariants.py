"""Forecast invariants: uncertainty present, non-negative density/uncertainty, deterministic queries."""

from __future__ import annotations


class InvariantViolation(AssertionError):
    pass


def assert_forecast_invariants(model, probe_points) -> None:
    """Queries each (time, lst, lat, alt_km) probe point twice and checks invariants."""
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
