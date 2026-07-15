"""Exercises each check-kind function against a fake in-memory ModelInterface."""

import numpy as np
import pytest

from rope_dev_tools.validation.checks import get_kind_function

pytest.importorskip("matplotlib")


class _FakeModel:
    def forecast(self, start, end):
        return {"window_start": start, "window_end": end}

    def query(self, time, lst, lat, alt_km):
        return {"density": 1.0e-12, "uncertainty": 1.0e-13}

    def query_grid(self, time, alt_km):
        return np.full((72, 36), 1.0e-12)


def test_lonlat_density_plot_via_model(tmp_path):
    fn = get_kind_function("lonlat_density_plot")
    output = fn(
        _FakeModel(), id="chk", time_point="2024-01-01 03:00:00", time_window_hours=6,
        altitudes_km=[400.0], out_dir=tmp_path,
    )
    assert "value" not in output  # plot-only kind
    assert (tmp_path / output["plots"][0]).is_file()


def test_lonlat_density_plot_raw_arrays_no_model(tmp_path):
    """Usable with no model, no suite, no runner."""
    fn = get_kind_function("lonlat_density_plot")
    density_2d = np.full((72, 36), 2.0e-12)
    output = fn(
        model=None, id="standalone", density_2d=density_2d, out_dir=tmp_path,
    )
    assert (tmp_path / output["plots"][0]).is_file()


def test_lonlat_density_plot_missing_fields_raises(tmp_path):
    fn = get_kind_function("lonlat_density_plot")
    with pytest.raises(ValueError):
        fn(model=None, id="bad", out_dir=tmp_path)  # neither density_2d nor model+time_point given


def test_rmse_timeseries_check(tmp_path):
    (tmp_path / "truth.csv").write_text(
        "datetime,lst,lat,alt_km,density\n2024-01-01 01:00:00,12.0,0.0,400.0,1.0e-12\n"
    )
    fn = get_kind_function("rmse_timeseries")
    output = fn(
        _FakeModel(), id="chk",
        start="2024-01-01 00:00:00", end="2024-01-01 03:00:00",
        truth_csv="truth.csv", threshold={"max": 1.0}, unit="kg/m3",
        out_dir=tmp_path, suite_dir=tmp_path,
    )
    assert output["value"] == pytest.approx(0.0)  # _FakeModel always matches the truth exactly
    assert output["passed"] is True


def test_satellite_lineout_check(tmp_path):
    (tmp_path / "track.csv").write_text(
        "datetime,lst,lat,alt_km,density\n"
        "2024-01-01 01:00:00,12.0,0.0,400.0,1.0e-12\n"
        "2024-01-01 02:00:00,6.0,10.0,410.0,1.0e-12\n"
    )
    fn = get_kind_function("satellite_lineout")
    output = fn(
        _FakeModel(), id="chk",
        start="2024-01-01 00:00:00", end="2024-01-01 03:00:00",
        satellite_track_csv="track.csv", threshold={"max": 1.0}, unit="kg/m3",
        out_dir=tmp_path, suite_dir=tmp_path,
    )
    assert output["value"] == pytest.approx(0.0)
    assert (tmp_path / output["plots"][0]).is_file()


def test_unknown_check_kind_raises():
    with pytest.raises(KeyError):
        get_kind_function("nonexistent_kind")
