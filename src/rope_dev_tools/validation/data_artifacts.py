"""Saved comparison-data helpers: validation_data/ sibling of plots/ in the exported dir."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def data_dir(out_dir) -> Path:
    d = Path(out_dir) / "validation_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_csv(out_dir, filename: str, df: "pd.DataFrame") -> str:
    df.to_csv(data_dir(out_dir) / filename, index=False)
    return f"validation_data/{filename}"


def load_csv(exported_dir, relative_path: str) -> "pd.DataFrame":
    path = Path(exported_dir) / relative_path
    has_datetime = "datetime" in pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, parse_dates=["datetime"] if has_datetime else None)


def save_npz(out_dir, filename: str, **arrays) -> str:
    np.savez_compressed(data_dir(out_dir) / filename, **arrays)
    return f"validation_data/{filename}"


def load_npz(exported_dir, relative_path: str) -> dict:
    with np.load(Path(exported_dir) / relative_path) as npz:
        return {k: npz[k] for k in npz.files}
