"""wam_ingest — builds physics_avg_csv/physics_model_hourly_npz from a suite, fetching each raw hourly timestamp at most once."""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rope_dev_tools.validation.time_utils import add_hours, format_time, parse_time, resolve_path
from rope_dev_tools.validation.truth_data import load_truth_csv
from rope_dev_tools.validation.wam_convert import read_wam_frame, read_wam_timesteps, sample_wam_frame
from rope_dev_tools.validation.wam_source import WamSourceGapError

_DEFAULT_MAX_CONCURRENT_FETCHES = 8


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


@dataclass
class TrackCsvTarget:
    output_filename: str
    source_satellite_track_csv: str   # guards against two periods claiming one output from different inputs
    sat: pd.DataFrame                 # loaded satellite_track_csv, original row order preserved
    floor: pd.Series                  # datetime64, per-row floor-hour bracket
    ceil: pd.Series                   # floor + 1h, uniformly -- even on exact-hour rows (weight becomes 0)
    weight: pd.Series                 # float in [0,1]
    before: np.ndarray                # (N,) float, NaN until the floor-side sample arrives
    after: np.ndarray                 # (N,) float, NaN until the ceil-side sample arrives
    timestamps: set = field(default_factory=set)   # set(floor) | set(ceil), as plain datetimes


def _target_key(target) -> tuple:
    """Dedup key across target types: (class name, output filename)."""
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
        altitudes_km = set(check["altitudes_km"])
        for period in check["periods"]:
            filename = period["physics_model_hourly_npz"]
            timestamps = _hourly_timestamps_inclusive(period["start"], period["horizon_hours"])

            target = targets.setdefault(filename, HourlyNpzTarget(filename))
            target.altitudes_km |= altitudes_km
            target.timestamps |= set(timestamps)
    return targets


def collect_track_csv_targets(checks: list, suite_dir) -> dict:
    """output_filename -> TrackCsvTarget, loading each referenced satellite_track_csv once."""
    targets: dict = {}
    for check in checks:
        if check["kind"] != "satellite_orbit_density":
            continue
        for period in check["periods"]:
            filename = period["physics_model_track_csv"]
            source_csv = period["satellite_track_csv"]

            existing = targets.get(filename)
            if existing is not None:
                if existing.source_satellite_track_csv != source_csv:
                    raise ValueError(
                        f"physics_model_track_csv {filename!r} referenced with conflicting "
                        f"satellite_track_csv sources: {existing.source_satellite_track_csv!r} "
                        f"vs {source_csv!r}"
                    )
                continue

            sat_path = resolve_path(suite_dir, source_csv)
            if not sat_path.is_file():
                raise ValueError(
                    f"{sat_path}: satellite_track_csv not found — run satellite ingestion first "
                    f"(scripts/build_satellite_data.py)"
                )
            sat = load_truth_csv(sat_path)
            if "lon" not in sat.columns:
                raise ValueError(f"{sat_path}: missing required column 'lon' for WAM along-track sampling")

            floor = sat["datetime"].dt.floor("h")
            ceil = floor + pd.Timedelta(hours=1)
            weight = (sat["datetime"] - floor) / pd.Timedelta(hours=1)
            timestamps = set(floor.dt.to_pydatetime()) | set(ceil.dt.to_pydatetime())

            targets[filename] = TrackCsvTarget(
                output_filename=filename, source_satellite_track_csv=source_csv, sat=sat,
                floor=floor, ceil=ceil, weight=weight,
                before=np.full(len(sat), np.nan), after=np.full(len(sat), np.nan),
                timestamps=timestamps,
            )
    return targets


def _write_avg_density_csv(rows: list, out_path: Path) -> None:
    """Writes accumulated {"datetime", "alt_km", "density"} rows, sorted, to out_path."""
    df = pd.DataFrame(rows).sort_values(["datetime", "alt_km"]).reset_index(drop=True)
    df["datetime"] = df["datetime"].apply(format_time)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _write_hourly_npz(timesteps: list, altitudes_km, out_path: Path) -> None:
    """Stacks accumulated WamTimesteps into physics_model_hourly_npz's (H, A, n_lon, n_lat) layout."""
    timesteps = sorted(timesteps, key=lambda ts: ts.time)
    altitudes_km = sorted(altitudes_km)
    first = timesteps[0]
    times = np.array([format_time(ts.time) for ts in timesteps])
    density = np.stack([
        np.stack([ts.lon_lat_density[alt_km] for alt_km in altitudes_km])
        for ts in timesteps
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, times=times, altitudes_km=np.array(altitudes_km, dtype=float),
        lon_values=first.lon_values, n_lat=first.n_lat,
        lat_min_deg=first.lat_min_deg, lat_max_deg=first.lat_max_deg, density=density,
    )


def _write_track_csv(target: TrackCsvTarget, out_path: Path) -> None:
    """Linearly interpolates target's before/after hourly samples to each satellite row's exact time."""
    if np.any(np.isnan(target.before)) or np.any(np.isnan(target.after)):
        raise AssertionError(
            f"{target.output_filename}: incomplete along-track interpolation data "
            f"(internal bug, not a data condition)"
        )
    weight = target.weight.to_numpy()
    density = (1 - weight) * target.before + weight * target.after
    df = pd.DataFrame({
        "datetime": target.sat["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lst": target.sat["lst"],
        "lat": target.sat["lat"],
        "alt_km": target.sat["alt_km"],
        "density": density,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _prefetch_timesteps(source, timestamps: list, scratch_dir: Path, max_workers: int):
    """Yields (timestamp, local_path) in order, keeping up to max_workers fetches in flight; local_path is None on a genuine gap."""
    if not timestamps:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = deque()
        it = iter(timestamps)
        for ts in it:
            pending.append((ts, executor.submit(source.fetch_timestep, ts, scratch_dir)))
            if len(pending) >= max_workers:
                break
        while pending:
            ts, future = pending.popleft()
            next_ts = next(it, None)
            if next_ts is not None:
                pending.append((next_ts, executor.submit(source.fetch_timestep, next_ts, scratch_dir)))
            try:
                local_path = future.result()
            except WamSourceGapError:
                local_path = None
            yield ts, local_path


def build(suite, out_dir, source, *, suite_dir=None, only_check_ids=None, progress=None,
          max_concurrent_fetches: int = _DEFAULT_MAX_CONCURRENT_FETCHES) -> list:
    """Fetches every timestamp any check needs at most once, writing each target once its data is complete."""
    checks = suite.checks
    if only_check_ids is not None:
        checks = [c for c in checks if c["id"] in only_check_ids]

    avg_targets = collect_avg_density_targets(checks)
    npz_targets = collect_hourly_npz_targets(checks)
    track_targets = {}
    if any(c["kind"] == "satellite_orbit_density" for c in checks):
        if suite_dir is None:
            raise ValueError(
                "suite has a satellite_orbit_density check but build() was not given suite_dir"
            )
        track_targets = collect_track_csv_targets(checks, suite_dir)

    all_targets = list(avg_targets.values()) + list(npz_targets.values()) + list(track_targets.values())
    if not all_targets:
        return []

    timestamp_index = defaultdict(list)
    remaining = {_target_key(t): set(t.timestamps) for t in all_targets}
    for t in all_targets:
        for ts in t.timestamps:
            timestamp_index[ts].append(t)

    accumulators = {_target_key(t): [] for t in all_targets}
    failed: dict = {}  # target key -> [missing timestamps]

    out_dir = Path(out_dir)
    scratch_dir = out_dir / ".wam_raw_scratch"
    written = []

    sorted_timestamps = sorted(timestamp_index)
    prefetched = _prefetch_timesteps(source, sorted_timestamps, scratch_dir, max_concurrent_fetches)
    for i, (ts, local_path) in enumerate(prefetched):
        if progress is not None:
            progress(i, len(sorted_timestamps), ts)

        dependents = timestamp_index[ts]

        if local_path is None:
            for t in dependents:
                key = _target_key(t)
                failed.setdefault(key, []).append(ts)
                remaining[key].discard(ts)
            continue

        scalar_dependents = [t for t in dependents if isinstance(t, (AvgDensityTarget, HourlyNpzTarget))]
        track_dependents = [t for t in dependents if isinstance(t, TrackCsvTarget)]

        if scalar_dependents:
            altitudes_needed = sorted({alt for t in scalar_dependents for alt in t.altitudes_km})
            timesteps = read_wam_timesteps(local_path, altitudes_km=altitudes_needed)
            if len(timesteps) != 1:
                raise ValueError(f"expected exactly one timestep in {local_path} for {ts}, found {len(timesteps)}")
            wam_ts = timesteps[0]
            if wam_ts.time.to_pydatetime() != ts:
                raise ValueError(
                    f"{local_path}: file's internal timestamp {wam_ts.time} does not match the "
                    f"requested hour {ts} — check filename_pattern/source configuration"
                )

            for t in scalar_dependents:
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

        if track_dependents:
            frames = read_wam_frame(local_path)
            if len(frames) != 1:
                raise ValueError(f"expected exactly one timestep in {local_path} for {ts}, found {len(frames)}")
            frame = frames[0]
            if frame.time.to_pydatetime() != ts:
                raise ValueError(
                    f"{local_path}: file's internal timestamp {frame.time} does not match the "
                    f"requested hour {ts} — check filename_pattern/source configuration"
                )

            for t in track_dependents:
                key = _target_key(t)
                for i in np.nonzero((t.floor == ts).to_numpy())[0]:
                    t.before[i] = sample_wam_frame(
                        frame.density, frame.lon_values, frame.lat_values, frame.alt_values,
                        lon=t.sat["lon"].iat[i], lat=t.sat["lat"].iat[i], alt_km=t.sat["alt_km"].iat[i],
                    )
                for i in np.nonzero((t.ceil == ts).to_numpy())[0]:
                    t.after[i] = sample_wam_frame(
                        frame.density, frame.lon_values, frame.lat_values, frame.alt_values,
                        lon=t.sat["lon"].iat[i], lat=t.sat["lat"].iat[i], alt_km=t.sat["alt_km"].iat[i],
                    )
                remaining[key].discard(ts)

        source.release(local_path)

        for t in dependents:
            key = _target_key(t)
            if remaining[key] or key in failed:
                continue  # not done, or permanently barred by a gap -- never write a partial file
            out_path = out_dir / t.output_filename
            if isinstance(t, AvgDensityTarget):
                _write_avg_density_csv(accumulators[key], out_path)
                del accumulators[key]
            elif isinstance(t, HourlyNpzTarget):
                _write_hourly_npz(accumulators[key], t.altitudes_km, out_path)
                del accumulators[key]
            else:
                _write_track_csv(t, out_path)
            written.append(out_path)

    if failed:
        detail = "\n".join(f"  {fn}: {sorted(str(ts) for ts in timestamps)}" for fn, timestamps in failed.items())
        raise ValueError(f"WAM ingestion: missing raw data (other targets completed):\n{detail}")

    return written
