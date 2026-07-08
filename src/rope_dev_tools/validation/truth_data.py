"""Loads dev-supplied truth-data CSVs referenced by a check's own path field
(e.g. rmse_timeseries's `truth_csv`, satellite_lineout's `satellite_track_csv`).

There is no shared "data_refs" indirection — each kind names its own path
field directly, resolved against the suite JSON's directory via
time_utils.resolve_path(). The CSV convention below (datetime, lst, lat,
alt_km, density[, uncertainty]) is a rope-dev-tools-local convention,
following the columns already used in .claude/docs/data-sources.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = {"datetime", "lst", "lat", "alt_km", "density"}


def load_truth_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return df
