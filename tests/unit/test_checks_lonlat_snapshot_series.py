"""lonlat_snapshot_series — one forecast feeds both static snapshots and a full-horizon animation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function

_GRID = {"n_lst": 8, "n_lat": 6, "lat_min_deg": -80.0, "lat_max_deg": 80.0}


class _FakeModel:
    grid = _GRID

    def forecast(self, start, end):
        return {"window_start": start, "window_end": end}

    def query_grid(self, time, alt_km):
        return np.full((8, 6), 1.0e-12)


def _write_physics_npz(path):
    np.savez(
        path,
        times=np.array(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
        n_lst=8, n_lat=6, lat_min_deg=-80.0, lat_max_deg=80.0,
        altitudes_km=np.array([400.0]),
        density=np.full((2, 1, 8, 6), 1.0e-12),
    )


def test_lonlat_snapshot_series_writes_snapshots_and_animation(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        start="2024-01-01 00:00:00", horizon_hours=1,
        days=["2024-01-01"], utc_hours=[0, 1], altitudes_km=[400.0],
        physics_model_hourly_npz="phys.npz",
    )

    assert len(output["plots"]) == 3  # physics snapshot, rope snapshot, animation
    for plot in output["plots"]:
        assert (tmp_path / plot).is_file()
    for data_path in output["data"]:
        assert (tmp_path / data_path).is_file()


def test_lonlat_snapshot_series_animation_over_72h_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
            start="2024-01-01 00:00:00", horizon_hours=73,
            days=["2024-01-01"], utc_hours=[0], altitudes_km=[400.0],
            physics_model_hourly_npz="phys.npz", include_animation=True,
        )


def test_lonlat_snapshot_series_missing_altitude_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
            start="2024-01-01 00:00:00", horizon_hours=1,
            days=["2024-01-01"], utc_hours=[0], altitudes_km=[900.0],
            physics_model_hourly_npz="phys.npz",
        )
