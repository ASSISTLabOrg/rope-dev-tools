"""Loads dev-supplied truth-data CSVs referenced by a check's own path field."""

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
