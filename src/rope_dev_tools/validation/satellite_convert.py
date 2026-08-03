"""satellite_convert — CDF day-file -> satellite_track_csv conversion primitives, network-agnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cdflib
except ImportError:  # pragma: no cover
    cdflib = None

# Confirmed against real GRACE and GRACE-FO CDF files (identical schema across both missions).
_DEFAULT_VARIABLE_NAMES = {
    "time": "time", "lon": "longitude", "lat": "latitude", "alt": "altitude",
    "lst": "local_solar_time", "density": "density", "validity_flag": "validity_flag",
}

_DEFAULT_CADENCE_SECONDS = 600


class SatelliteVariableNotFoundError(ValueError):
    pass


def _require_cdflib() -> None:
    if cdflib is None:
        raise ImportError(
            "cdflib is required for satellite conversion; install rope-dev-tools[satellite]"
        )


def _resolve_names(available: set, variable_names: "dict | None") -> dict:
    variable_names = variable_names or {}
    resolved = {}
    for field, default in _DEFAULT_VARIABLE_NAMES.items():
        name = variable_names.get(field, default)
        if name not in available:
            raise SatelliteVariableNotFoundError(
                f"could not find a CDF variable named {name!r} for {field!r}; "
                f"available variables: {sorted(available)}. Pass "
                f"variable_names={{{field!r}: 'actual_name'}} to override."
            )
        resolved[field] = name
    return resolved


def _circular_mean(values: np.ndarray, period: float) -> float:
    """Mean of a periodic quantity"""
    angles = values * (2.0 * np.pi / period)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return float((mean_angle * period / (2.0 * np.pi)) % period)


_EPOCH = pd.Timestamp("1970-01-01")


def _downsample_track(df: pd.DataFrame, cadence_seconds: int) -> pd.DataFrame:
    """Absolute-UTC-aligned bin averaging"""
    bin_idx = (df["datetime"] - _EPOCH) // pd.Timedelta(seconds=cadence_seconds)
    g = df.groupby(bin_idx)
    return pd.DataFrame({
        "datetime": g["datetime"].mean(),
        "lst": g["lst"].apply(lambda s: _circular_mean(s.to_numpy(), 24.0)),
        "lat": g["lat"].mean(),
        "lon": g["lon"].apply(lambda s: _circular_mean(s.to_numpy(), 360.0)),
        "alt_km": g["alt_km"].mean(),
        "density": g["density"].mean(),
    }).reset_index(drop=True)


def read_satellite_day(cdf_path, *, variable_names=None, include_flagged=False,
                        cadence_seconds=_DEFAULT_CADENCE_SECONDS) -> pd.DataFrame:
    """One raw day-file -> DataFrame(datetime, lst, lat, lon, alt_km, density)."""
    _require_cdflib()
    with cdflib.cdfread.CDF(str(cdf_path)) as reader:
        available = set(reader.cdf_info().zVariables)
        names = _resolve_names(available, variable_names)

        time = cdflib.cdfepoch.to_datetime(reader.varget(names["time"]))
        df = pd.DataFrame({
            "datetime": pd.to_datetime(time),
            "lst": reader.varget(names["lst"]).astype(float),
            "lat": reader.varget(names["lat"]).astype(float),
            "lon": reader.varget(names["lon"]).astype(float) % 360.0,
            "alt_km": reader.varget(names["alt"]).astype(float) / 1000.0,
            "density": reader.varget(names["density"]).astype(float),
        })
        if not include_flagged:
            flag = reader.varget(names["validity_flag"])
            df = df[np.asarray(flag) == 0].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{cdf_path}: no rows remain after validity_flag filtering")

    if cadence_seconds:
        df = _downsample_track(df, cadence_seconds)
    return df.sort_values("datetime").reset_index(drop=True)


def convert_satellite_track_csv(cdf_paths, out_csv_path, *, variable_names=None, include_flagged=False,
                                 cadence_seconds=_DEFAULT_CADENCE_SECONDS) -> Path:
    """read_satellite_day() over 1+ local day files, write satellite_track_csv"""
    paths = [cdf_paths] if isinstance(cdf_paths, (str, Path)) else list(cdf_paths)
    frames = [
        read_satellite_day(p, variable_names=variable_names, include_flagged=include_flagged,
                            cadence_seconds=cadence_seconds)
        for p in paths
    ]
    df = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path
