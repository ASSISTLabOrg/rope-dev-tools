"""wam_convert primitives against small synthetic .nc fixtures with hand-computable expected values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from rope_dev_tools.validation.wam_convert import (
    WamVariableNotFoundError,
    _linear_bracket,
    _periodic_bracket,
    convert_avg_density_csv,
    convert_hourly_npz,
    read_wam_frame,
    read_wam_timesteps,
    sample_wam_frame,
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


def test_read_wam_timesteps_grid_mean_constant_density(tmp_path):
    _write_nc(tmp_path / "a.nc", time="2024-01-01T00:00:00",
              density_fn=lambda a, i, j: 10.0 * (a + 1))

    timesteps = read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[100.0, 200.0])
    assert len(timesteps) == 1
    ts = timesteps[0]
    assert ts.time == pd.Timestamp("2024-01-01T00:00:00")
    assert ts.grid_mean_density[100.0] == pytest.approx(10.0)
    assert ts.grid_mean_density[200.0] == pytest.approx(20.0)
    assert ts.n_lon == _N_LON
    assert ts.n_lat == _N_LAT
    assert ts.lat_min_deg == pytest.approx(-60.0)
    assert ts.lat_max_deg == pytest.approx(60.0)


def test_read_wam_timesteps_keeps_native_longitude_order(tmp_path):
    # Not rolled/shifted to LST -- density should come back in the exact same lon-index order
    # the raw file declares it in, regardless of the timestep's own UTC time.
    _write_nc(tmp_path / "a.nc", time="2024-01-01T06:00:00",
              density_fn=lambda a, i, j: float(i))

    ts = read_wam_timesteps(tmp_path / "a.nc", altitudes_km=[100.0])[0]
    lon_profile = ts.lon_lat_density[100.0][:, 0]
    assert np.array_equal(lon_profile, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(ts.lon_values, np.arange(_N_LON) * (360.0 / _N_LON))
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
    assert ts.lon_lat_density[100.0].shape == (_N_LON, _N_LAT)


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
        assert np.array_equal(npz["lon_values"], np.arange(_N_LON) * (360.0 / _N_LON))
        assert npz["n_lat"] == _N_LAT
        assert np.allclose(npz["density"][0, 0], 10.0)
        assert np.allclose(npz["density"][1, 1], 60.0)


def test_read_wam_frame_shape_and_values(tmp_path):
    _write_nc(tmp_path / "a.nc", time="2024-01-01T00:00:00", density_fn=lambda a, i, j: 10.0 * (a + 1) + i + j)

    frames = read_wam_frame(tmp_path / "a.nc")
    assert len(frames) == 1
    f = frames[0]
    assert f.time == pd.Timestamp("2024-01-01T00:00:00")
    assert f.density.shape == (len(_ALTITUDES), _N_LON, _N_LAT)
    assert f.lon_values.shape == (_N_LON,)
    assert f.lat_values.shape == (_N_LAT,)
    assert f.alt_values.shape == (len(_ALTITUDES),)
    assert f.density[0, 0, 0] == pytest.approx(10.0)
    assert f.density[1, 2, 1] == pytest.approx(20.0 + 2 + 1)


def test_periodic_bracket_wraps_and_interior():
    lon_values = np.array([0.0, 90.0, 180.0, 270.0])
    assert _periodic_bracket(lon_values, 315.0) == (3, 0, pytest.approx(0.5))
    assert _periodic_bracket(lon_values, 45.0) == (0, 1, pytest.approx(0.5))
    assert _periodic_bracket(lon_values, 0.0) == (0, 1, pytest.approx(0.0))


def test_periodic_bracket_rejects_non_uniform_spacing():
    with pytest.raises(ValueError):
        _periodic_bracket(np.array([0.0, 10.0, 180.0, 270.0]), 45.0)


def test_linear_bracket_exact_endpoints_and_interior():
    values = np.array([100.0, 200.0, 300.0])
    assert _linear_bracket(values, 100.0, label="x") == (0, 1, pytest.approx(0.0))
    assert _linear_bracket(values, 300.0, label="x") == (1, 2, pytest.approx(1.0))
    assert _linear_bracket(values, 150.0, label="x") == (0, 1, pytest.approx(0.5))


def test_linear_bracket_out_of_range_raises():
    values = np.array([100.0, 200.0])
    with pytest.raises(ValueError, match="latitude"):
        _linear_bracket(values, 50.0, label="latitude")
    with pytest.raises(ValueError, match="latitude"):
        _linear_bracket(values, 250.0, label="latitude")


def test_linear_bracket_single_value_axis():
    assert _linear_bracket(np.array([100.0]), 100.0, label="x") == (0, 0, 0.0)


def test_sample_wam_frame_exact_grid_point_and_interior():
    # additive gradient (alt + lon + lat) -> trilinear interpolation recovers it exactly
    lon_values = np.array([0.0, 90.0, 180.0, 270.0])
    lat_values = np.array([0.0, 50.0, 100.0])
    alt_values = np.array([100.0, 200.0])
    density = np.zeros((2, 4, 3))
    for a_idx, a in enumerate(alt_values):
        for lo_idx, lo in enumerate(lon_values):
            for la_idx, la in enumerate(lat_values):
                density[a_idx, lo_idx, la_idx] = a + lo + la

    assert sample_wam_frame(density, lon_values, lat_values, alt_values,
                             lon=90.0, lat=50.0, alt_km=100.0) == pytest.approx(240.0)
    assert sample_wam_frame(density, lon_values, lat_values, alt_values,
                             lon=45.0, lat=25.0, alt_km=150.0) == pytest.approx(220.0)


def test_sample_wam_frame_periodic_wrap():
    lon_values = np.array([0.0, 90.0, 180.0, 270.0])
    lat_values = np.array([0.0, 50.0])
    alt_values = np.array([100.0])
    density = np.zeros((1, 4, 2))
    for lo_idx in range(4):
        density[0, lo_idx, :] = float(lo_idx)

    v = sample_wam_frame(density, lon_values, lat_values, alt_values, lon=315.0, lat=0.0, alt_km=100.0)
    assert v == pytest.approx(1.5)


def test_sample_wam_frame_out_of_range_raises():
    lon_values = np.array([0.0, 90.0, 180.0, 270.0])
    lat_values = np.array([0.0, 100.0])
    alt_values = np.array([100.0, 200.0])
    density = np.zeros((2, 4, 2))

    with pytest.raises(ValueError):
        sample_wam_frame(density, lon_values, lat_values, alt_values, lon=0.0, lat=-10.0, alt_km=100.0)
    with pytest.raises(ValueError):
        sample_wam_frame(density, lon_values, lat_values, alt_values, lon=0.0, lat=0.0, alt_km=50.0)
