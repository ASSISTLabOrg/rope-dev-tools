"""wam_convert — NetCDF(.nc) -> physics_avg_csv/physics_model_hourly_npz conversion primitives, AWS-agnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rope_dev_tools.validation.variable_names import resolve_variable_names

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

_DEFAULT_VARIABLE_NAMES = {"time": "time", "lon": "lon", "lat": "lat", "alt": "hlevs", "density": "den"}


class WamVariableNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class WamTimestep:
    time: "pd.Timestamp"
    grid_mean_density: dict     # alt_km -> float, plain mean over the full lon x lat grid
    lon_lat_density: dict       # alt_km -> (n_lon, n_lat) np.ndarray, native geographic longitude order
    lon_values: np.ndarray      # actual longitude (deg) of each lon_lat_density row
    n_lon: int
    n_lat: int
    lat_min_deg: float
    lat_max_deg: float


@dataclass(frozen=True)
class WamFrame:
    """One raw WAM timestep's full, unreduced 3D density field"""
    time: "pd.Timestamp"
    lon_values: np.ndarray
    lat_values: np.ndarray
    alt_values: np.ndarray
    density: np.ndarray   # (n_alt, n_lon, n_lat)


def _require_xarray() -> None:
    """Raises ImportError with an install hint if xarray wasn't importable."""
    if xr is None:
        raise ImportError(
            "xarray (and a NetCDF backend, e.g. netCDF4) is required for WAM conversion; "
            "install rope-dev-tools[wam]"
        )


def _resolve_names(ds, variable_names: "dict | None") -> dict:
    """Maps each of _DEFAULT_VARIABLE_NAMES's fields to its actual name in ds, override or default."""
    available = set(ds.variables) | set(ds.dims)
    return resolve_variable_names(
        available, variable_names, _DEFAULT_VARIABLE_NAMES, error_cls=WamVariableNotFoundError,
        noun="NetCDF variable/dimension", available_noun="variables/dims",
    )


def _select_altitude_indices(alt_values, altitudes_km, *, atol: float = 1e-2) -> dict:
    """alt_km -> index of the closest matching alt_values entry; raises if none is close enough."""
    alt_values = np.asarray(alt_values, dtype=float)
    indices = {}
    for alt_km in altitudes_km:
        matches = np.nonzero(np.isclose(alt_values, alt_km, atol=atol))[0]
        if len(matches) == 0:
            raise ValueError(
                f"altitude {alt_km} km not present in source data; available altitudes: "
                f"{sorted(float(a) for a in alt_values)}"
            )
        indices[alt_km] = int(matches[0])
    return indices


def read_wam_timesteps(nc_path, *, altitudes_km, variable_names=None) -> list:
    """Reads every timestep in one raw WAM .nc file."""
    _require_xarray()
    with xr.open_dataset(nc_path) as ds:
        names = _resolve_names(ds, variable_names)
        alt_indices = _select_altitude_indices(ds[names["alt"]].values, altitudes_km)

        lon_values = np.asarray(ds[names["lon"]].values, dtype=float)
        lat_values = ds[names["lat"]].values
        n_lon, n_lat = lon_values.size, lat_values.size
        lat_min_deg, lat_max_deg = float(lat_values.min()), float(lat_values.max())

        density = ds[names["density"]]
        time_dim = names["time"]
        raw_times = ds[time_dim].values
        times = list(pd.to_datetime(np.atleast_1d(raw_times)))

        timesteps = []
        for t_idx, t in enumerate(times):
            frame = density.isel({time_dim: t_idx}) if time_dim in density.dims else density

            grid_mean, lon_lat = {}, {}
            for alt_km, alt_idx in alt_indices.items():
                alt_frame = frame.isel({names["alt"]: alt_idx}).transpose(names["lon"], names["lat"])
                values = alt_frame.values  # (n_lon, n_lat)
                grid_mean[alt_km] = float(np.mean(values))
                lon_lat[alt_km] = values

            timesteps.append(WamTimestep(
                time=t, grid_mean_density=grid_mean, lon_lat_density=lon_lat, lon_values=lon_values,
                n_lon=n_lon, n_lat=n_lat, lat_min_deg=lat_min_deg, lat_max_deg=lat_max_deg,
            ))
        return timesteps


def read_wam_frame(nc_path, *, variable_names=None) -> list:
    """Reads every timestep's full, unreduced 3D density field from one raw WAM .nc file."""
    _require_xarray()
    with xr.open_dataset(nc_path) as ds:
        names = _resolve_names(ds, variable_names)
        lon_values = np.asarray(ds[names["lon"]].values, dtype=float)
        lat_values = np.asarray(ds[names["lat"]].values, dtype=float)
        alt_values = np.asarray(ds[names["alt"]].values, dtype=float)

        density = ds[names["density"]]
        time_dim = names["time"]
        raw_times = ds[time_dim].values
        times = list(pd.to_datetime(np.atleast_1d(raw_times)))

        frames = []
        for t_idx, t in enumerate(times):
            frame = density.isel({time_dim: t_idx}) if time_dim in density.dims else density
            frame = frame.transpose(names["alt"], names["lon"], names["lat"])
            frames.append(WamFrame(
                time=t, lon_values=lon_values, lat_values=lat_values, alt_values=alt_values,
                density=frame.values,
            ))
        return frames


def _periodic_bracket(values, query, period: float = 360.0) -> tuple:
    """(i0, i1, weight) bracketing query in a uniformly-spaced, wraparound (e.g. longitude) axis."""
    values = np.asarray(values, dtype=float)
    n = values.size
    spacing = period / n
    if not np.allclose(np.diff(values), spacing, atol=1e-3):
        raise ValueError(f"axis is not uniformly spaced in {spacing} steps; cannot bracket periodically")
    frac_idx = ((query % period) - values[0]) / spacing
    i0 = int(np.floor(frac_idx)) % n
    i1 = (i0 + 1) % n
    weight = float(frac_idx - np.floor(frac_idx))
    return i0, i1, weight


def _linear_bracket(values, query, *, label: str) -> tuple:
    """(i0, i1, weight) bracketing query in a sorted, non-wraparound axis; raises if out of range."""
    values = np.asarray(values, dtype=float)
    if query < values[0] or query > values[-1]:
        raise ValueError(f"{label} {query} is outside the covered range [{values[0]}, {values[-1]}]")
    if values.size == 1:
        return 0, 0, 0.0
    i1 = int(np.searchsorted(values, query, side="right"))
    i1 = max(1, min(i1, values.size - 1))
    i0 = i1 - 1
    span = values[i1] - values[i0]
    weight = 0.0 if span == 0 else float((query - values[i0]) / span)
    return i0, i1, weight


def sample_wam_frame(density, lon_values, lat_values, alt_values, lon, lat, alt_km) -> float:
    """Trilinear interpolation of a WamFrame's density at (lon, lat, alt_km); lon wraps periodically."""
    lon_i0, lon_i1, lon_w = _periodic_bracket(lon_values, lon)
    lat_i0, lat_i1, lat_w = _linear_bracket(lat_values, lat, label="latitude")
    alt_i0, alt_i1, alt_w = _linear_bracket(alt_values, alt_km, label="alt_km")

    def _at_alt(alt_idx):
        v00 = density[alt_idx, lon_i0, lat_i0]
        v01 = density[alt_idx, lon_i0, lat_i1]
        v10 = density[alt_idx, lon_i1, lat_i0]
        v11 = density[alt_idx, lon_i1, lat_i1]
        v0 = v00 * (1 - lat_w) + v01 * lat_w
        v1 = v10 * (1 - lat_w) + v11 * lat_w
        return v0 * (1 - lon_w) + v1 * lon_w

    return float(_at_alt(alt_i0) * (1 - alt_w) + _at_alt(alt_i1) * alt_w)


def convert_avg_density_csv(nc_paths, out_csv_path, *, altitudes_km, variable_names=None) -> Path:
    """Grid-mean density per altitude -> physics_avg_csv (datetime, alt_km, density)."""
    paths = [nc_paths] if isinstance(nc_paths, (str, Path)) else list(nc_paths)
    rows = []
    for p in paths:
        for ts in read_wam_timesteps(p, altitudes_km=altitudes_km, variable_names=variable_names):
            for alt_km, value in ts.grid_mean_density.items():
                rows.append({"datetime": ts.time, "alt_km": alt_km, "density": value})
    if not rows:
        raise ValueError(f"no timesteps read from {paths!r}")

    df = pd.DataFrame(rows).sort_values(["datetime", "alt_km"]).reset_index(drop=True)
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path


def convert_hourly_npz(nc_paths, out_npz_path, *, altitudes_km, variable_names=None) -> Path:
    """Full (H, A, n_lon, n_lat) lon/lat density grids per hour -> physics_model_hourly_npz."""
    paths = [nc_paths] if isinstance(nc_paths, (str, Path)) else list(nc_paths)
    all_timesteps = []
    for p in paths:
        all_timesteps.extend(read_wam_timesteps(p, altitudes_km=altitudes_km, variable_names=variable_names))
    if not all_timesteps:
        raise ValueError(f"no timesteps read from {paths!r}")
    all_timesteps.sort(key=lambda ts: ts.time)

    first = all_timesteps[0]
    times = np.array([ts.time.strftime("%Y-%m-%d %H:%M:%S") for ts in all_timesteps])
    density = np.stack([
        np.stack([ts.lon_lat_density[alt_km] for alt_km in altitudes_km])
        for ts in all_timesteps
    ])  # (H, A, n_lon, n_lat)

    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        times=times,
        altitudes_km=np.array(list(altitudes_km), dtype=float),
        lon_values=first.lon_values,
        n_lat=first.n_lat,
        lat_min_deg=first.lat_min_deg,
        lat_max_deg=first.lat_max_deg,
        density=density,
    )
    return out_npz_path
