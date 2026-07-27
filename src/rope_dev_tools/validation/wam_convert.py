"""wam_convert — NetCDF(.nc) -> physics_avg_csv/physics_model_hourly_npz conversion primitives, AWS-agnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

# lon is geographic longitude, not LST — see _lst_roll_shift().
_DEFAULT_VARIABLE_NAMES = {"time": "time", "lon": "lon", "lat": "lat", "alt": "hlevs", "density": "den"}


class WamVariableNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class WamTimestep:
    time: "pd.Timestamp"
    grid_mean_density: dict     # alt_km -> float, plain mean over the full lon x lat grid
    lst_lat_density: dict       # alt_km -> (n_lst, n_lat) np.ndarray, longitude rolled into LST order
    n_lst: int
    n_lat: int
    lat_min_deg: float
    lat_max_deg: float


def _require_xarray() -> None:
    if xr is None:
        raise ImportError(
            "xarray (and a NetCDF backend, e.g. netCDF4) is required for WAM conversion; "
            "install rope-dev-tools[wam]"
        )


def _resolve_names(ds, variable_names: "dict | None") -> dict:
    variable_names = variable_names or {}
    available = set(ds.variables) | set(ds.dims)
    resolved = {}
    for field, default in _DEFAULT_VARIABLE_NAMES.items():
        name = variable_names.get(field, default)
        if name not in available:
            raise WamVariableNotFoundError(
                f"could not find a NetCDF variable/dimension named {name!r} for {field!r}; "
                f"available variables/dims: {sorted(available)}. Pass "
                f"variable_names={{{field!r}: 'actual_name'}} to override."
            )
        resolved[field] = name
    return resolved


def _select_altitude_indices(alt_values, altitudes_km, *, atol: float = 1e-2) -> dict:
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


def _lst_roll_shift(lon_values, utc_hour: float) -> int:
    """Bin-index shift converting a uniformly-spaced ascending lon axis into LST order at utc_hour."""
    lon_values = np.asarray(lon_values, dtype=float)
    n_lon = lon_values.size
    spacing = 360.0 / n_lon
    if not np.allclose(np.diff(lon_values), spacing, atol=1e-3):
        raise ValueError(
            f"longitude axis is not uniformly spaced in {spacing} deg steps; cannot convert to "
            f"LST via a circular shift"
        )
    base_shift = (utc_hour + lon_values[0] / 15.0) / (24.0 / n_lon)
    return int(round(base_shift)) % n_lon


def read_wam_timesteps(nc_path, *, altitudes_km, variable_names=None) -> list:
    """Reads every timestep in one raw WAM .nc file."""
    _require_xarray()
    with xr.open_dataset(nc_path) as ds:
        names = _resolve_names(ds, variable_names)
        alt_indices = _select_altitude_indices(ds[names["alt"]].values, altitudes_km)

        lon_values = ds[names["lon"]].values
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
            utc_hour = t.hour + t.minute / 60.0 + t.second / 3600.0
            shift = _lst_roll_shift(lon_values, utc_hour)

            grid_mean, lst_lat = {}, {}
            for alt_km, alt_idx in alt_indices.items():
                alt_frame = frame.isel({names["alt"]: alt_idx}).transpose(names["lon"], names["lat"])
                values = alt_frame.values  # (n_lon, n_lat)
                grid_mean[alt_km] = float(np.mean(values))
                lst_lat[alt_km] = np.roll(values, shift=shift, axis=0)

            timesteps.append(WamTimestep(
                time=t, grid_mean_density=grid_mean, lst_lat_density=lst_lat,
                n_lst=n_lon, n_lat=n_lat, lat_min_deg=lat_min_deg, lat_max_deg=lat_max_deg,
            ))
        return timesteps


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
    """Full (H, A, n_lst, n_lat) density grid -> physics_model_hourly_npz (times, altitudes_km, n_lst, n_lat, lat_min_deg, lat_max_deg, density)."""
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
        np.stack([ts.lst_lat_density[alt_km] for alt_km in altitudes_km])
        for ts in all_timesteps
    ])  # (H, A, n_lst, n_lat)

    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        times=times,
        altitudes_km=np.array(list(altitudes_km), dtype=float),
        n_lst=first.n_lst,
        n_lat=first.n_lat,
        lat_min_deg=first.lat_min_deg,
        lat_max_deg=first.lat_max_deg,
        density=density,
    )
    return out_npz_path
