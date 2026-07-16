"""doy_lat_orbit_density — doy vs lat grids, ascending/descending x satellite/physics/rope."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function


class _FakeModel:
    def forecast(self, start, end):
        return {"window_start": start, "window_end": end}

    def query(self, time, lst, lat, alt_km):
        return {"density": 1.0e-12}


def _write_satellite_csv(path):
    path.write_text(
        "datetime,lst,lat,alt_km,density,ascending\n"
        "2024-01-01 00:00:00,12.0,-10.0,400.0,1.0e-12,1\n"
        "2024-01-01 01:00:00,12.0,10.0,400.0,1.1e-12,1\n"
        "2024-01-02 00:00:00,12.0,10.0,400.0,1.2e-12,0\n"
        "2024-01-02 01:00:00,12.0,-10.0,400.0,1.3e-12,0\n"
    )


def _write_physics_csv(path):
    path.write_text(
        "datetime,lst,lat,alt_km,density\n"
        "2024-01-01 00:00:00,12.0,-10.0,400.0,0.9e-12\n"
        "2024-01-01 01:00:00,12.0,10.0,400.0,1.0e-12\n"
        "2024-01-02 00:00:00,12.0,10.0,400.0,1.1e-12\n"
        "2024-01-02 01:00:00,12.0,-10.0,400.0,1.2e-12\n"
    )


def test_doy_lat_orbit_density_writes_six_plots_per_altitude(tmp_path):
    _write_satellite_csv(tmp_path / "sat.csv")
    _write_physics_csv(tmp_path / "phys.csv")
    fn = get_kind_function("doy_lat_orbit_density")

    output = fn(
        _FakeModel(), id="doy_test", out_dir=tmp_path, suite_dir=tmp_path,
        start="2024-01-01 00:00:00", end="2024-01-03 00:00:00",
        satellite_track_csv="sat.csv", physics_model_track_csv="phys.csv",
        altitudes_km=[400.0], lat_bin_deg=10.0,
    )

    assert len(output["plots"]) == 6
    for plot in output["plots"]:
        assert (tmp_path / plot).is_file()
    assert (tmp_path / output["data"][0]).is_file()
    assert "statistics" not in output


def test_doy_lat_orbit_density_requires_ascending_column(tmp_path):
    (tmp_path / "sat.csv").write_text(
        "datetime,lst,lat,alt_km,density\n2024-01-01 00:00:00,12.0,-10.0,400.0,1.0e-12\n"
    )
    _write_physics_csv(tmp_path / "phys.csv")
    fn = get_kind_function("doy_lat_orbit_density")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="doy_test", out_dir=tmp_path, suite_dir=tmp_path,
            start="2024-01-01 00:00:00", end="2024-01-03 00:00:00",
            satellite_track_csv="sat.csv", physics_model_track_csv="phys.csv",
            altitudes_km=[400.0], lat_bin_deg=10.0,
        )
