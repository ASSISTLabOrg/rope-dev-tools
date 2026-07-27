"""wam_convert primitives against small synthetic .nc fixtures with hand-computable expected values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from rope_dev_tools.validation.wam_convert import (
    WamVariableNotFoundError,
    _lst_roll_shift,
    convert_avg_density_csv,
    convert_hourly_npz,
    read_wam_timesteps,
)

_N_LON, _N_LAT = 4, 3
_ALTITUDES = (100.0, 200.0)


def _write_nc(path, *, time, density_fn, density_name="den", dim_order=("time", "hlevs", "lat", "lon")):
    """density_fn(alt_idx, lon_idx, lat_idx) -> float, so expected values are hand-computable."""
    lon = np.arange(_N_LON) * (360.0 / _N_LON)
    lat = np.linspace(-60.0, 60.0, _N_LAT)
    hlevs = np.array(_ALTITUDES, dtype=np.float32)

    den = np.zeros((1, len(_ALTITUDES), _N_LAT, _N_LON), dtype=np.float32)
    for a in range(len(_ALTITUDES)):
        for j in range(_N_LAT):
            for i in range(_N_LON):
                den[0, a, j, i] = density_fn(a, i, j)

    ds = xr.Dataset(
        {density_name: (("time", "hlevs", "lat", "lon"), den)},
        coords={"time": [np.datetime64(time)], "hlevs": hlevs, "lat": lat, "lon": lon},
    )
    if dim_order != ("time", "hlevs", "lat", "lon"):
        ds[density_name] = ds[density_name].transpose(*dim_order)
    ds.to_netcdf(path)


def test_lst_roll_shift_matches_hand_computation():
    lon_values = np.array([0.0, 90.0, 180.0, 270.0])
    assert _lst_roll_shift(lon_values, utc_hour=6.0) == 1
    assert _lst_roll_shift(lon_values, utc_hour=0.0) == 0
    assert _lst_roll_shift(lon_values, utc_hour=12.0) == 2


def test_lst_roll_shift_rejects_non_uniform_spacing():
    with pytest.raises(ValueError):
        _lst_roll_shift(np.array([0.0, 10.0, 180.0, 270.0]), utc_hour=6.0)


def test_read_wam_timesteps_grid_mean_constant_density(tmp_path):
    _write_nc(tmp_path / "a.nc", time="2024-01-01T00:00:00",
              density_fn=lambda a, i, j: 10.0 * (a + 1))

    timesteps = read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[100.0, 200.0])
    assert len(timesteps) == 1
    ts = timesteps[0]
    assert ts.time == pd.Timestamp("2024-01-01T00:00:00")
    assert ts.grid_mean_density[100.0] == pytest.approx(10.0)
    assert ts.grid_mean_density[200.0] == pytest.approx(20.0)
    assert ts.n_lst == _N_LON
    assert ts.n_lat == _N_LAT
    assert ts.lat_min_deg == pytest.approx(-60.0)
    assert ts.lat_max_deg == pytest.approx(60.0)


def test_read_wam_timesteps_lst_shift_reorders_lon_gradient(tmp_path):
    _write_nc(tmp_path / "a.nc", time="2024-01-01T06:00:00",
              density_fn=lambda a, i, j: float(i))

    ts = read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[100.0])[0]
    lst_profile = ts.lst_lat_density[100.0][:, 0]
    assert np.array_equal(lst_profile, np.roll(np.array([0.0, 1.0, 2.0, 3.0]), shift=1))
    assert ts.grid_mean_density[100.0] == pytest.approx(np.mean([0.0, 1.0, 2.0, 3.0]))


def test_read_wam_timesteps_missing_altitude_raises(tmp_path):
    _write_nc(tmp_path / "a.nc", time="2024-01-01T00:00:00", density_fn=lambda a, i, j: 1.0)
    with pytest.raises(ValueError, match="900"):
        read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[900.0])


def test_read_wam_timesteps_variable_name_override(tmp_path):
    _write_nc(tmp_path / "a.nc", time="2024-01-01T00:00:00",
              density_fn=lambda a, i, j: 5.0, density_name="foo_density")

    with pytest.raises(WamVariableNotFoundError):
        read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[100.0])

    timesteps = read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[100.0],
                                    variable_names={"density": "foo_density"})
    assert timesteps[0].grid_mean_density[100.0] == pytest.approx(5.0)


def test_read_wam_timesteps_dimension_order_does_not_matter(tmp_path):
    _write_nc(tmp_path / "a.nc", time="2024-01-01T00:00:00",
              density_fn=lambda a, i, j: 10.0 * (a + 1),
              dim_order=("hlevs", "time", "lon", "lat"))

    ts = read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[100.0, 200.0])[0]
    assert ts.grid_mean_density[100.0] == pytest.approx(10.0)
    assert ts.grid_mean_density[200.0] == pytest.approx(20.0)
    assert ts.lst_lat_density[100.0].shape == (_N_LON, _N_LAT)


def test_convert_avg_density_csv_multiple_files_concatenates_sorted(tmp_path):
    _write_nc(tmp_path / "t1.nc", time="2024-01-01T01:00:00", density_fn=lambda a, i, j: 10.0 * (a + 1))
    _write_nc(tmp_path / "t0.nc", time="2024-01-01T00:00:00", density_fn=lambda a, i, j: 20.0 * (a + 1))

    out_path = convert_avg_density_csv(
        [tmp_path / "t1.nc", tmp_path / "t0.nc"], tmp_path / "out.csv", altitudes_km=[100.0, 200.0],
    )
    df = pd.read_csv(out_path, parse_dates=["datetime"])
    assert list(df["datetime"]) == [pd.Timestamp("2024-01-01T00:00:00")] * 2 + [pd.Timestamp("2024-01-01T01:00:00")] * 2
    row0 = df[(df["datetime"] == pd.Timestamp("2024-01-01T00:00:00")) & (df["alt_km"] == 100.0)]
    assert row0["density"].iloc[0] == pytest.approx(20.0)


def test_convert_hourly_npz_shape_and_values(tmp_path):
    _write_nc(tmp_path / "t0.nc", time="2024-01-01T00:00:00", density_fn=lambda a, i, j: 10.0 * (a + 1))
    _write_nc(tmp_path / "t1.nc", time="2024-01-01T01:00:00", density_fn=lambda a, i, j: 30.0 * (a + 1))

    out_path = convert_hourly_npz(
        [tmp_path / "t0.nc", tmp_path / "t1.nc"], tmp_path / "out.npz", altitudes_km=[100.0, 200.0],
    )
    with np.load(out_path) as npz:
        assert list(npz["times"]) == ["2024-01-01 00:00:00", "2024-01-01 01:00:00"]
        assert npz["density"].shape == (2, 2, _N_LON, _N_LAT)
        assert npz["n_lst"] == _N_LON
        assert npz["n_lat"] == _N_LAT
        assert np.allclose(npz["density"][0, 0], 10.0)
        assert np.allclose(npz["density"][1, 1], 60.0)
