"""wam_ingest — builds physics_avg_csv/physics_model_hourly_npz from a suite, fetching each raw hourly timestamp at most once."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rope_dev_tools.validation.time_utils import add_hours, format_time, parse_time
from rope_dev_tools.validation.wam_convert import read_wam_timesteps


@dataclass
class AvgDensityTarget:
    output_filename: str
    altitudes_km: set = field(default_factory=set)
    timestamps: set = field(default_factory=set)


@dataclass
class HourlyNpzTarget:
    output_filename: str
    altitudes_km: set = field(default_factory=set)
    timestamps: set = field(default_factory=set)


def _target_key(target) -> tuple:
    return (type(target).__name__, target.output_filename)


def _hourly_timestamps(start: str, end: str) -> list:
    """Every hour in the half-open interval [start, end), as on-the-hour datetimes."""
    t0, t1 = parse_time(start), parse_time(end)
    if t0.minute or t0.second or t0.microsecond:
        raise ValueError(f"start {start!r} is not on an exact hour boundary")
    if t1.minute or t1.second or t1.microsecond:
        raise ValueError(f"end {end!r} is not on an exact hour boundary")
    if t1 <= t0:
        raise ValueError(f"end {end!r} is not after start {start!r}")
    out, t = [], t0
    while t < t1:
        out.append(t)
        t += timedelta(hours=1)
    return out


def _hourly_timestamps_inclusive(start: str, horizon_hours) -> list:
    """H = horizon_hours + 1 hourly steps starting at start, inclusive of both endpoints."""
    t0 = parse_time(start)
    if t0.minute or t0.second or t0.microsecond:
        raise ValueError(f"start {start!r} is not on an exact hour boundary")
    if horizon_hours < 0 or int(horizon_hours) != horizon_hours:
        raise ValueError(f"horizon_hours must be a non-negative integer, got {horizon_hours!r}")
    return [t0 + timedelta(hours=h) for h in range(int(horizon_hours) + 1)]


def collect_avg_density_targets(checks: list) -> dict:
    """output_filename -> AvgDensityTarget, altitudes/timestamps unioned across every reference."""
    targets: dict = {}
    for check in checks:
        if check["kind"] != "avg_density_vs_time":
            continue
        altitudes_km = set(check["altitudes_km"])
        for period in check["periods"]:
            raw = period["physics_avg_csv"]
            filenames = raw if isinstance(raw, list) else [raw]
            timestamps = _hourly_timestamps(period["start"], period["end"])
            years = sorted({t.year for t in timestamps})
            if len(filenames) != len(years):
                raise ValueError(
                    f"check {check['id']!r} period {period['label']!r}: physics_avg_csv has "
                    f"{len(filenames)} entr{'y' if len(filenames) == 1 else 'ies'} but "
                    f"[{period['start']}, {period['end']}) spans {len(years)} calendar year(s) "
                    f"{years}; provide exactly one filename per spanned year, in ascending order"
                )
            timestamps_by_year = defaultdict(list)
            for t in timestamps:
                timestamps_by_year[t.year].append(t)

            for year, filename in zip(years, filenames):
                target = targets.setdefault(filename, AvgDensityTarget(filename))
                target.altitudes_km |= altitudes_km
                target.timestamps |= set(timestamps_by_year[year])
    return targets


def collect_hourly_npz_targets(checks: list) -> dict:
    """output_filename -> HourlyNpzTarget, altitudes/timestamps unioned across every reference."""
    targets: dict = {}
    for check in checks:
        if check["kind"] != "lonlat_snapshot_series":
            continue
        filename = check["physics_model_hourly_npz"]
        timestamps = _hourly_timestamps_inclusive(check["start"], check["horizon_hours"])

        target = targets.setdefault(filename, HourlyNpzTarget(filename))
        target.altitudes_km |= set(check["altitudes_km"])
        target.timestamps |= set(timestamps)
    return targets


def _write_avg_density_csv(rows: list, out_path: Path) -> None:
    df = pd.DataFrame(rows).sort_values(["datetime", "alt_km"]).reset_index(drop=True)
    df["datetime"] = df["datetime"].apply(format_time)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _write_hourly_npz(timesteps: list, altitudes_km, out_path: Path) -> None:
    timesteps = sorted(timesteps, key=lambda ts: ts.time)
    altitudes_km = sorted(altitudes_km)
    first = timesteps[0]
    times = np.array([format_time(ts.time) for ts in timesteps])
    density = np.stack([
        np.stack([ts.lst_lat_density[alt_km] for alt_km in altitudes_km])
        for ts in timesteps
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, times=times, altitudes_km=np.array(altitudes_km, dtype=float),
        n_lst=first.n_lst, n_lat=first.n_lat,
        lat_min_deg=first.lat_min_deg, lat_max_deg=first.lat_max_deg, density=density,
    )


def build(suite, out_dir, source, *, only_check_ids=None) -> list:
    """Fetches/converts every truth-data artifact the suite's checks need. Returns written paths."""
    checks = suite.checks
    if only_check_ids is not None:
        checks = [c for c in checks if c["id"] in only_check_ids]

    avg_targets = collect_avg_density_targets(checks)
    npz_targets = collect_hourly_npz_targets(checks)
    all_targets = list(avg_targets.values()) + list(npz_targets.values())
    if not all_targets:
        return []

    timestamp_index = defaultdict(list)
    remaining = {_target_key(t): set(t.timestamps) for t in all_targets}
    for t in all_targets:
        for ts in t.timestamps:
            timestamp_index[ts].append(t)

    accumulators = {_target_key(t): [] for t in all_targets}

    out_dir = Path(out_dir)
    scratch_dir = out_dir / ".wam_raw_scratch"
    written = []

    for ts in sorted(timestamp_index):
        dependents = timestamp_index[ts]
        altitudes_needed = sorted({alt for t in dependents for alt in t.altitudes_km})

        local_path = source.fetch_timestep(ts, scratch_dir)
        timesteps = read_wam_timesteps(local_path, altitudes_km=altitudes_needed)
        if len(timesteps) != 1:
            raise ValueError(f"expected exactly one timestep in {local_path} for {ts}, found {len(timesteps)}")
        wam_ts = timesteps[0]
        if wam_ts.time.to_pydatetime() != ts:
            raise ValueError(
                f"{local_path}: file's internal timestamp {wam_ts.time} does not match the "
                f"requested hour {ts} — check filename_pattern/source configuration"
            )

        for t in dependents:
            key = _target_key(t)
            if isinstance(t, AvgDensityTarget):
                for alt_km in t.altitudes_km:
                    accumulators[key].append({
                        "datetime": wam_ts.time, "alt_km": alt_km,
                        "density": wam_ts.grid_mean_density[alt_km],
                    })
            else:
                accumulators[key].append(wam_ts)
            remaining[key].discard(ts)

        source.release(local_path)

        for t in dependents:
            key = _target_key(t)
            if remaining[key]:
                continue
            out_path = out_dir / t.output_filename
            if isinstance(t, AvgDensityTarget):
                _write_avg_density_csv(accumulators[key], out_path)
            else:
                _write_hourly_npz(accumulators[key], t.altitudes_km, out_path)
            written.append(out_path)
            del accumulators[key]

    return written
