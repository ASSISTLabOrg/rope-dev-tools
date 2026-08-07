"""doy_lat_orbit_density — doy vs lat grids, ascending/descending x satellite/physics/rope."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function


class _FakeModel:
    def __init__(self, density=1.0e-12, uncert=1.0e-13):
        self.compute_uncertainty_calls = []
        self._density = density
        self._uncert = uncert

    def forecast(self, start, end, *, compute_uncertainty=False):
        self.compute_uncertainty_calls.append(compute_uncertainty)
        return {"window_start": start, "window_end": end}

    def query(self, time, lst, lat, alt_km):
        return {"density": self._density, "uncertainty": self._uncert}


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


def test_doy_lat_orbit_density_passes_compute_uncertainty_through_to_forecast(tmp_path):
    _write_satellite_csv(tmp_path / "sat.csv")
    _write_physics_csv(tmp_path / "phys.csv")
    fn = get_kind_function("doy_lat_orbit_density")
    model = _FakeModel(density=1.5e-12)

    fn(
        model, id="doy_test", out_dir=tmp_path, suite_dir=tmp_path,
        start="2024-01-01 00:00:00", end="2024-01-03 00:00:00",
        satellite_track_csv="sat.csv", physics_model_track_csv="phys.csv",
        altitudes_km=[400.0], lat_bin_deg=10.0, uncertainty=True,
    )
    assert model.compute_uncertainty_calls == [True]


def test_doy_lat_orbit_density_computes_uncertainty_only_for_rope_vs_satellite(tmp_path):
    _write_satellite_csv(tmp_path / "sat.csv")
    _write_physics_csv(tmp_path / "phys.csv")
    fn = get_kind_function("doy_lat_orbit_density")

    output = fn(
        # rope density (1.5e-12) differs from satellite/physics so rmse/std stay nonzero.
        _FakeModel(density=1.5e-12), id="doy_test", out_dir=tmp_path, suite_dir=tmp_path,
        start="2024-01-01 00:00:00", end="2024-01-03 00:00:00",
        satellite_track_csv="sat.csv", physics_model_track_csv="phys.csv",
        altitudes_km=[400.0], lat_bin_deg=10.0, statistics=["bias", "rmse"], uncertainty=True,
    )

    for direction in ("ascending", "descending"):
        entry = output["statistics"]["400.0km"][direction]
        assert entry["rope_vs_satellite_uncertainty"].keys() == {"bias", "rmse"}
        assert "physics_vs_satellite_uncertainty" not in entry


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
