"""satellite_orbit_density — satellite/physics/rope density along a track, RMSE against satellite."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function


class _FakeModel:
    def forecast(self, start, end):
        return {"window_start": start, "window_end": end}

    def query(self, time, lst, lat, alt_km):
        return {"density": 1.0e-12, "uncertainty": 1.0e-13}


def _write_track_csv(path, density):
    path.write_text(
        "datetime,lst,lat,alt_km,density\n"
        f"2024-01-01 00:00:00,12.0,0.0,400.0,{density}\n"
        f"2024-01-01 01:00:00,13.0,5.0,400.0,{density}\n"
    )


def test_satellite_orbit_density_writes_plot_data_and_value(tmp_path):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        start="2024-01-01 00:00:00", end="2024-01-01 02:00:00",
        satellite_track_csv="sat.csv", physics_model_track_csv="phys.csv",
    )

    assert (tmp_path / output["plots"][0]).is_file()
    assert (tmp_path / output["data"][0]).is_file()
    assert output["value"] == pytest.approx(0.0, abs=1e-20)
    assert "statistics" not in output


def test_satellite_orbit_density_row_mismatch_raises(tmp_path):
    (tmp_path / "sat.csv").write_text(
        "datetime,lst,lat,alt_km,density\n2024-01-01 00:00:00,12.0,0.0,400.0,1.0e-12\n"
    )
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
            start="2024-01-01 00:00:00", end="2024-01-01 02:00:00",
            satellite_track_csv="sat.csv", physics_model_track_csv="phys.csv",
        )


def test_satellite_orbit_density_computes_requested_statistics(tmp_path):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        start="2024-01-01 00:00:00", end="2024-01-01 02:00:00",
        satellite_track_csv="sat.csv", physics_model_track_csv="phys.csv",
        statistics=["bias"],
    )

    assert set(output["statistics"]) == {"rope_vs_satellite", "physics_vs_satellite"}
