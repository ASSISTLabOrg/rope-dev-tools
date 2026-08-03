"""satellite_ingest — builds satellite_track_csv from a suite, fetching each raw day at most once.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd

from rope_dev_tools.validation.satellite_convert import read_satellite_day
from rope_dev_tools.validation.satellite_source import SatelliteSourceGapError
from rope_dev_tools.validation.time_utils import parse_time


@dataclass
class SatelliteTrackTarget:
    output_filename: str
    dates: set = field(default_factory=set)


def _target_key(target: SatelliteTrackTarget) -> str:
    return target.output_filename


def _calendar_days(start: str, end: str) -> list:
    """Every date d with [d, d+1day) overlapping the half-open interval [start, end)."""
    t0, t1 = parse_time(start), parse_time(end)
    if t1 <= t0:
        raise ValueError(f"end {end!r} is not after start {start!r}")
    last_day = (t1 - timedelta(seconds=1)).date()
    out, d = [], t0.date()
    while d <= last_day:
        out.append(d)
        d += timedelta(days=1)
    return out


def collect_satellite_track_targets(checks: list) -> dict:
    """output_filename -> SatelliteTrackTarget"""
    targets: dict = {}
    for check in checks:
        if check["kind"] != "satellite_orbit_density":
            continue
        for period in check["periods"]:
            filename = period["satellite_track_csv"]
            target = targets.setdefault(filename, SatelliteTrackTarget(filename))
            target.dates |= set(_calendar_days(period["start"], period["end"]))
    return targets


def build(suite, out_dir, source, *, only_check_ids=None, cadence_seconds=600, progress=None) -> list:
    """Fetches/converts every satellite_track_csv the suite's checks need"""
    checks = suite.checks
    if only_check_ids is not None:
        checks = [c for c in checks if c["id"] in only_check_ids]

    targets = collect_satellite_track_targets(checks)
    if not targets:
        return []

    date_index = defaultdict(list)
    remaining = {_target_key(t): set(t.dates) for t in targets.values()}
    for t in targets.values():
        for d in t.dates:
            date_index[d].append(t)

    accumulators = {_target_key(t): [] for t in targets.values()}
    failed: dict = {}  # output_filename -> [missing dates]

    out_dir = Path(out_dir)
    scratch_dir = out_dir / ".satellite_raw_scratch"
    written = []

    sorted_dates = sorted(date_index)
    for i, d in enumerate(sorted_dates):
        if progress is not None:
            progress(i, len(sorted_dates), d)

        dependents = date_index[d]

        try:
            local_path = source.fetch_day(d, scratch_dir)
        except SatelliteSourceGapError:
            for t in dependents:
                failed.setdefault(_target_key(t), []).append(d)
                remaining[_target_key(t)].discard(d)
            continue

        try:
            day_df = read_satellite_day(local_path, cadence_seconds=cadence_seconds)
        finally:
            source.release(local_path)

        for t in dependents:
            accumulators[_target_key(t)].append(day_df)
            remaining[_target_key(t)].discard(d)

        for t in dependents:
            key = _target_key(t)
            if remaining[key] or key in failed:
                continue  # not done, or permanently barred by a gap -- never write a partial file
            out_path = out_dir / t.output_filename
            df = pd.concat(accumulators.pop(key), ignore_index=True).sort_values("datetime").reset_index(drop=True)
            df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            written.append(out_path)

    if failed:
        detail = "\n".join(f"  {fn}: {sorted(str(dd) for dd in dates)}" for fn, dates in failed.items())
        raise ValueError(f"satellite ingestion: missing raw data (other targets completed):\n{detail}")

    return written
