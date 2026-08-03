"""Time/path helpers shared by check-kind functions and model interfaces."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path


def parse_time(t: str) -> datetime:
    t = t.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError(f"cannot parse time string: {t!r}")


def format_time(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def add_hours(t: str, hours: float) -> str:
    return format_time(parse_time(t) + timedelta(hours=hours))


def hours_between(start: str, end: str) -> int:
    delta = parse_time(end) - parse_time(start)
    return max(1, math.ceil(delta.total_seconds() / 3600))


def hourly_range(start: str, end: str, interval_hours: float = 1) -> list:
    """List of ISO-ish time strings from start to end, inclusive, stepped by interval_hours."""
    t0, t1 = parse_time(start), parse_time(end)
    step = timedelta(hours=interval_hours)
    out = []
    t = t0
    while t <= t1:
        out.append(format_time(t))
        t += step
    return out


def resolve_start_delta(start: str, end: str, delta_hours) -> tuple:
    """(forecast_start, query_start_dt) for one start_delta against a period's fixed [start, end]
    evaluation window"""
    start_dt, end_dt = parse_time(start), parse_time(end)
    forecast_start = add_hours(start, delta_hours)
    forecast_start_dt = parse_time(forecast_start)
    if forecast_start_dt >= end_dt:
        raise ValueError(
            f"start_delta {delta_hours!r}h shifts start to {forecast_start!r}, leaving no time "
            f"before end {end!r}"
        )
    query_start_dt = max(start_dt, forecast_start_dt)
    return forecast_start, query_start_dt


def lst_values_for(lon_values, t: str):
    dt = parse_time(t)
    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return (utc_hour + lon_values / 15.0) % 24.0


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolves a possibly-relative path field against base_dir."""
    p = Path(value)
    return p if p.is_absolute() else Path(base_dir) / p
