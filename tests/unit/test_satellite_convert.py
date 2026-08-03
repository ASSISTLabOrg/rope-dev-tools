"""satellite_convert: CDF -> satellite_track_csv, against small synthetic CDF fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

cdflib = pytest.importorskip("cdflib")

from rope_dev_tools.validation.satellite_convert import (
    SatelliteVariableNotFoundError,
    _circular_mean,
    _downsample_track,
    convert_satellite_track_csv,
    read_satellite_day,
)


def _write_cdf(path, *, times, lon, lat, alt_m, lst, density, validity_flag=None, density_name="density"):
    n = len(times)
    if validity_flag is None:
        validity_flag = [0] * n
    epoch_values = cdflib.cdfepoch.compute_epoch(
        [[t.year, t.month, t.day, t.hour, t.minute, t.second, 0, 0, 0] for t in times]
    )

    writer = cdflib.cdfwrite.CDF(str(path))
    writer.write_globalattrs({})

    def _write(name, data_type, values):
        writer.write_var(
            {"Variable": name, "Data_Type": data_type, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_data=np.asarray(values),
        )

    _write("time", 31, epoch_values)
    _write("longitude", 22, np.asarray(lon, dtype=np.float64))
    _write("latitude", 22, np.asarray(lat, dtype=np.float64))
    _write("altitude", 22, np.asarray(alt_m, dtype=np.float64))
    _write("local_solar_time", 22, np.asarray(lst, dtype=np.float64))
    _write(density_name, 22, np.asarray(density, dtype=np.float64))
    _write("validity_flag", 1, np.asarray(validity_flag, dtype=np.int8))
    writer.close()


def test_read_satellite_day_basic(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 10)]
    _write_cdf(tmp_path / "day.cdf", times=times, lon=[10.0, 20.0], lat=[1.0, 2.0],
               alt_m=[500000.0, 500100.0], lst=[12.0, 12.1], density=[1e-12, 1.1e-12])

    df = read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0)
    assert len(df) == 2
    assert df["alt_km"].iloc[0] == pytest.approx(500.0)
    assert df["lon"].iloc[0] == pytest.approx(10.0)
    assert list(df.columns) == ["datetime", "lst", "lat", "lon", "alt_km", "density"]


def test_read_satellite_day_normalizes_longitude(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 10)]
    _write_cdf(tmp_path / "day.cdf", times=times, lon=[-179.0, -1.0], lat=[0.0, 0.0],
               alt_m=[500000.0, 500000.0], lst=[12.0, 12.0], density=[1.0, 1.0])

    df = read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0)
    assert df["lon"].iloc[0] == pytest.approx(181.0)
    assert df["lon"].iloc[1] == pytest.approx(359.0)


def test_read_satellite_day_drops_flagged_rows_by_default(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 10), datetime(2024, 1, 1, 0, 0, 20)]
    _write_cdf(tmp_path / "day.cdf", times=times, lon=[0.0] * 3, lat=[0.0] * 3, alt_m=[500000.0] * 3,
               lst=[12.0] * 3, density=[1.0, 2.0, 3.0], validity_flag=[0, 1, 0])

    df = read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0)
    assert len(df) == 2
    assert set(df["density"]) == {1.0, 3.0}

    df_all = read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0, include_flagged=True)
    assert len(df_all) == 3


def test_read_satellite_day_all_flagged_raises(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 0)]
    _write_cdf(tmp_path / "day.cdf", times=times, lon=[0.0], lat=[0.0], alt_m=[500000.0], lst=[12.0],
               density=[1.0], validity_flag=[1])

    with pytest.raises(ValueError):
        read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0)


def test_read_satellite_day_variable_name_override(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 0)]
    _write_cdf(tmp_path / "day.cdf", times=times, lon=[0.0], lat=[0.0], alt_m=[500000.0], lst=[12.0],
               density=[5.0], density_name="foo_density")

    with pytest.raises(SatelliteVariableNotFoundError):
        read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0)

    df = read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0, variable_names={"density": "foo_density"})
    assert df["density"].iloc[0] == pytest.approx(5.0)


def test_read_satellite_day_cadence_zero_keeps_all_rows(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 0) + timedelta(seconds=i * 10) for i in range(20)]
    _write_cdf(tmp_path / "day.cdf", times=times, lon=[0.0] * 20, lat=[0.0] * 20, alt_m=[500000.0] * 20,
               lst=[12.0] * 20, density=[1.0] * 20)

    df = read_satellite_day(tmp_path / "day.cdf", cadence_seconds=0)
    assert len(df) == 20


def test_downsample_track_bin_entirely_flagged_is_omitted(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 10), datetime(2024, 1, 1, 0, 10, 0)]
    _write_cdf(tmp_path / "day.cdf", times=times, lon=[0.0] * 3, lat=[0.0] * 3, alt_m=[500000.0] * 3,
               lst=[12.0] * 3, density=[1.0, 2.0, 3.0], validity_flag=[1, 1, 0])

    df = read_satellite_day(tmp_path / "day.cdf", cadence_seconds=600)
    assert len(df) == 1
    assert df["density"].iloc[0] == pytest.approx(3.0)


def test_convert_satellite_track_csv_multiple_files_sorted(tmp_path):
    _write_cdf(tmp_path / "day2.cdf", times=[datetime(2024, 1, 2, 0, 0, 0)],
               lon=[0.0], lat=[0.0], alt_m=[500000.0], lst=[12.0], density=[2.0])
    _write_cdf(tmp_path / "day1.cdf", times=[datetime(2024, 1, 1, 0, 0, 0)],
               lon=[0.0], lat=[0.0], alt_m=[500000.0], lst=[12.0], density=[1.0])

    out = convert_satellite_track_csv([tmp_path / "day2.cdf", tmp_path / "day1.cdf"], tmp_path / "out.csv",
                                       cadence_seconds=0)
    df = pd.read_csv(out)
    assert list(df["density"]) == [1.0, 2.0]


def _assert_angle_approx(actual, expected, period, tol=1e-6):
    """0 and 360 (or any two values a whole period apart) are the same angle."""
    diff = abs(actual - expected) % period
    assert min(diff, period - diff) < tol, f"{actual} != {expected} (mod {period})"


def test_circular_mean_wraps_correctly():
    _assert_angle_approx(_circular_mean(np.array([359.0, 1.0]), 360.0), 0.0, 360.0)
    assert _circular_mean(np.array([10.0, 20.0]), 360.0) == pytest.approx(15.0, abs=1e-6)


def test_downsample_track_hand_computed():
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:10", "2024-01-01 00:10:00"]),
        "lst": [23.0, 1.0, 12.0],
        "lat": [0.0, 2.0, 10.0],
        "lon": [350.0, 10.0, 100.0],
        "alt_km": [500.0, 502.0, 510.0],
        "density": [1.0, 2.0, 3.0],
    })

    out = _downsample_track(df, cadence_seconds=600)
    assert len(out) == 2
    assert out["datetime"].iloc[0] == pd.Timestamp("2024-01-01 00:00:05")
    assert out["density"].iloc[0] == pytest.approx(1.5)
    assert out["lat"].iloc[0] == pytest.approx(1.0)
    _assert_angle_approx(out["lon"].iloc[0], 0.0, 360.0)
    _assert_angle_approx(out["lst"].iloc[0], 0.0, 24.0)
    assert out["density"].iloc[1] == pytest.approx(3.0)
