"""Time/path helpers shared by check-kind functions and model interfaces.

Kept as plain, public-named utilities (not buried as private helpers in one
module) since multiple independent check-kind functions need them, and kinds
otherwise don't share any code with each other.
"""

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
    """List of ISO-ish time strings from start to end (inclusive), stepped
    by interval_hours."""
    t0, t1 = parse_time(start), parse_time(end)
    step = timedelta(hours=interval_hours)
    out = []
    t = t0
    while t <= t1:
        out.append(format_time(t))
        t += step
    return out


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolves a possibly-relative path field against base_dir (typically
    the suite JSON's own directory) -- any check-kind function needing a
    file reference can call this for that field, without a schema-level
    convention forcing every kind to name it the same thing."""
    p = Path(value)
    return p if p.is_absolute() else Path(base_dir) / p
