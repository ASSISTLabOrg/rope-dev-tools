"""Loads dev-supplied truth-data CSVs referenced by a check's own path field."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = {"datetime", "lst", "lat", "alt_km", "density"}
_AVG_DENSITY_COLUMNS = {"datetime", "alt_km", "density"}


def load_truth_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return df


def load_avg_density_csv(path) -> pd.DataFrame:
    """Grid-average density vs time at fixed altitudes — datetime, alt_km, density[, uncertainty]. path: CSV path or list of paths."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    frames = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["datetime"])
        missing = _AVG_DENSITY_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{p}: missing required columns {sorted(missing)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def load_ascending_track_csv(path: Path) -> pd.DataFrame:
    """load_truth_csv plus a required 'ascending' (0/1) orbit-direction column."""
    df = load_truth_csv(path)
    if "ascending" not in df.columns:
        raise ValueError(f"{path}: missing required column 'ascending'")
    return df
