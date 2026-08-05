"""wam_ingest: target collection (dedup/union across checks) and the streaming fetch-once build loop."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rope_dev_tools.validation.schema_types import ValidationSuite
from rope_dev_tools.validation.wam_convert import WamFrame, WamTimestep
from rope_dev_tools.validation.wam_ingest import (
    _hourly_timestamps,
    _hourly_timestamps_inclusive,
    _prefetch_timesteps,
    build,
    collect_avg_density_targets,
    collect_hourly_npz_targets,
    collect_track_csv_targets,
)
from rope_dev_tools.validation.wam_source import WamRawDataSource, WamSourceGapError


def test_hourly_timestamps_half_open_excludes_end():
    ts = _hourly_timestamps("2001-01-01 00:00:00", "2002-01-01 00:00:00")
    assert ts[0] == datetime(2001, 1, 1, 0)
    assert ts[-1] == datetime(2001, 12, 31, 23)
    assert len(ts) == 365 * 24


def test_hourly_timestamps_multi_year_span():
    ts = _hourly_timestamps("2008-12-01 00:00:00", "2019-12-01 00:00:00")
    assert {t.year for t in ts} == set(range(2008, 2020))


def test_hourly_timestamps_rejects_non_hour_boundary():
    with pytest.raises(ValueError):
        _hourly_timestamps("2001-01-01 00:30:00", "2001-01-02 00:00:00")


def test_hourly_timestamps_inclusive_is_horizon_plus_one():
    ts = _hourly_timestamps_inclusive("2023-01-01 00:00:00", 72)
    assert len(ts) == 73
    assert ts[0] == datetime(2023, 1, 1, 0)
    assert ts[-1] == datetime(2023, 1, 4, 0)


def _avg_check(check_id, periods, altitudes_km=(300.0,)):
    return {"id": check_id, "kind": "avg_density_vs_time", "periods": periods, "altitudes_km": list(altitudes_km)}


def test_collect_avg_density_targets_single_file_period():
    checks = [_avg_check("c1", [{"label": "2013", "start": "2013-01-01 00:00:00",
                                  "end": "2014-01-01 00:00:00", "physics_avg_csv": "wam_2013.csv"}])]
    targets = collect_avg_density_targets(checks)
    assert set(targets) == {"wam_2013.csv"}
    assert targets["wam_2013.csv"].altitudes_km == {300.0}
    assert len(targets["wam_2013.csv"].timestamps) == 365 * 24


def test_collect_avg_density_targets_multi_file_list_length_mismatch_raises():
    checks = [_avg_check("c1", [{
        "label": "solar cycle", "start": "2008-12-01 00:00:00", "end": "2019-12-01 00:00:00",
        "physics_avg_csv": ["wam_2008.csv", "wam_2009.csv"],  # should be 12 entries, not 2
    }])]
    with pytest.raises(ValueError, match="12"):
        collect_avg_density_targets(checks)


def test_collect_avg_density_targets_merges_overlapping_hour_across_checks():
    storm = _avg_check("storms", [{"label": "March 2013", "start": "2013-03-15 00:00:00",
                                    "end": "2013-03-20 00:00:00", "physics_avg_csv": "wam_2013.csv"}],
                        altitudes_km=(250.0,))
    yearly = _avg_check("yearly", [{"label": "2013", "start": "2013-01-01 00:00:00",
                                     "end": "2014-01-01 00:00:00", "physics_avg_csv": "wam_2013.csv"}],
                         altitudes_km=(400.0,))
    targets = collect_avg_density_targets([storm, yearly])
    assert set(targets) == {"wam_2013.csv"}
    assert targets["wam_2013.csv"].altitudes_km == {250.0, 400.0}
    assert len(targets["wam_2013.csv"].timestamps) == 365 * 24


def _lonlat_check(check_id, periods, altitudes_km=(300.0,)):
    return {"id": check_id, "kind": "lonlat_snapshot_series", "periods": periods,
            "altitudes_km": list(altitudes_km)}


def test_collect_hourly_npz_targets_unions_altitudes_for_shared_filename():
    check_a = _lonlat_check("a", [{"label": "p1", "start": "2023-01-01 00:00:00", "horizon_hours": 2,
                                    "utc_hours": [0], "physics_model_hourly_npz": "shared.npz"}],
                             altitudes_km=(300.0,))
    check_b = _lonlat_check("b", [{"label": "p1", "start": "2023-01-01 00:00:00", "horizon_hours": 2,
                                    "utc_hours": [0], "physics_model_hourly_npz": "shared.npz"}],
                             altitudes_km=(400.0,))
    targets = collect_hourly_npz_targets([check_a, check_b])
    assert set(targets) == {"shared.npz"}
    assert targets["shared.npz"].altitudes_km == {300.0, 400.0}
    assert len(targets["shared.npz"].timestamps) == 3


def test_collect_hourly_npz_targets_unions_across_periods_in_one_check():
    check = _lonlat_check("a", [
        {"label": "p1", "start": "2023-01-01 00:00:00", "horizon_hours": 1,
         "utc_hours": [0], "physics_model_hourly_npz": "shared.npz"},
        {"label": "p2", "start": "2023-01-03 00:00:00", "horizon_hours": 1,
         "utc_hours": [0], "physics_model_hourly_npz": "shared.npz"},
    ])
    targets = collect_hourly_npz_targets([check])
    assert set(targets) == {"shared.npz"}
    # 2 timestamps per period (horizon_hours=1 -> inclusive of both endpoints), non-overlapping days
    assert len(targets["shared.npz"].timestamps) == 4


class _FakeSource(WamRawDataSource):
    """Records every fetch/release call; hands back a marker path per timestamp (no real .nc file)."""

    def __init__(self):
        self.fetches = []
        self.releases = []

    def fetch_timestep(self, dt, scratch_dir):
        self.fetches.append(dt)
        return dt  # stands in for a path; read_wam_timesteps is monkeypatched to accept it

    def release(self, path):
        self.releases.append(path)


class _ConcurrencyTrackingSource(WamRawDataSource):
    """Sleeps briefly on every fetch and records the high-water mark of simultaneously in-flight
    fetches, so the prefetch pool's bound (and that it's actually achieving concurrency, not
    accidentally serializing) can both be asserted directly."""

    def __init__(self, delay: float = 0.02):
        self._delay = delay
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def fetch_timestep(self, dt, scratch_dir):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        time.sleep(self._delay)
        with self._lock:
            self._in_flight -= 1
        return dt

    def release(self, path):
        return None


class _FailingSource(WamRawDataSource):
    def fetch_timestep(self, dt, scratch_dir):
        raise RuntimeError(f"boom: {dt}")

    def release(self, path):
        return None


class _GapSource(WamRawDataSource):
    """Raises WamSourceGapError for the given timestamps, else behaves like _FakeSource."""

    def __init__(self, gap_timestamps):
        self._gap_timestamps = set(gap_timestamps)
        self.fetches = []
        self.releases = []

    def fetch_timestep(self, dt, scratch_dir):
        self.fetches.append(dt)
        if dt in self._gap_timestamps:
            raise WamSourceGapError(dt, [dt.year])
        return dt

    def release(self, path):
        self.releases.append(path)


def test_prefetch_timesteps_empty_input_yields_nothing():
    assert list(_prefetch_timesteps(_FakeSource(), [], Path("/tmp"), max_workers=4)) == []


def test_prefetch_timesteps_preserves_order_regardless_of_completion_order():
    timestamps = [datetime(2023, 1, 1, h) for h in range(10)]
    source = _ConcurrencyTrackingSource(delay=0.01)

    results = list(_prefetch_timesteps(source, timestamps, Path("/tmp"), max_workers=3))

    assert [ts for ts, _ in results] == timestamps
    assert [path for _, path in results] == timestamps  # _FakeSource-style: fetch_timestep returns dt


def test_prefetch_timesteps_bounds_and_achieves_concurrency():
    timestamps = [datetime(2023, 1, 1, h) for h in range(10)]
    source = _ConcurrencyTrackingSource(delay=0.03)

    list(_prefetch_timesteps(source, timestamps, Path("/tmp"), max_workers=3))

    assert source.max_in_flight <= 3  # the prefetch window is never exceeded
    assert source.max_in_flight >= 2  # and it's genuinely overlapping fetches, not accidentally serial


def test_prefetch_timesteps_propagates_fetch_errors():
    with pytest.raises(RuntimeError, match="boom"):
        list(_prefetch_timesteps(_FailingSource(), [datetime(2023, 1, 1)], Path("/tmp"), max_workers=2))


def test_prefetch_timesteps_yields_none_on_gap_instead_of_raising():
    ts = datetime(2023, 1, 1)
    results = list(_prefetch_timesteps(_GapSource([ts]), [ts], Path("/tmp"), max_workers=2))
    assert results == [(ts, None)]


def test_build_with_max_concurrent_fetches_one_matches_default(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)

    check = _avg_check("c1", [{"label": "p", "start": "2013-03-15 00:00:00",
                                "end": "2013-03-15 05:00:00", "physics_avg_csv": "c1.csv"}])
    suite = ValidationSuite(1, 1, [check])

    written_serial = build(suite, tmp_path / "serial", _FakeSource(), max_concurrent_fetches=1)
    written_default = build(suite, tmp_path / "default", _FakeSource())

    df_serial = pd.read_csv(tmp_path / "serial" / "c1.csv")
    df_default = pd.read_csv(tmp_path / "default" / "c1.csv")
    pd.testing.assert_frame_equal(df_serial, df_default)
    assert {p.name for p in written_serial} == {p.name for p in written_default}


def _fake_read_wam_timesteps(dt, *, altitudes_km, variable_names=None):
    grid = np.zeros((2, 2))
    return [WamTimestep(
        time=pd.Timestamp(dt),
        grid_mean_density={alt: float(dt.hour) + alt for alt in altitudes_km},
        lon_lat_density={alt: grid for alt in altitudes_km},
        lon_values=np.array([0.0, 180.0]),
        n_lon=2, n_lat=2, lat_min_deg=-10.0, lat_max_deg=10.0,
    )]


def test_build_fetches_shared_timestamp_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)

    storm = _avg_check("storms", [{"label": "March 2013", "start": "2013-03-15 00:00:00",
                                    "end": "2013-03-15 02:00:00", "physics_avg_csv": "wam_2013.csv"}],
                        altitudes_km=(250.0,))
    yearly = _avg_check("yearly", [{"label": "2013 span", "start": "2013-03-15 00:00:00",
                                     "end": "2013-03-15 03:00:00", "physics_avg_csv": "wam_2013.csv"}],
                         altitudes_km=(400.0,))
    lonlat = _lonlat_check("snap", [{"label": "p1", "start": "2013-03-15 00:00:00", "horizon_hours": 1,
                                      "utc_hours": [0], "physics_model_hourly_npz": "wam_2013_snap.npz"}])

    suite = ValidationSuite(1, 1, [storm, yearly, lonlat])
    source = _FakeSource()
    written = build(suite, tmp_path, source)

    # hour 2013-03-15 00:00 is shared by all three checks -> fetched exactly once
    assert source.fetches.count(datetime(2013, 3, 15, 0)) == 1
    # hour 01:00 shared by storm+yearly -> fetched exactly once
    assert source.fetches.count(datetime(2013, 3, 15, 1)) == 1
    # every fetch was released
    assert len(source.fetches) == len(source.releases)

    written_names = {p.name for p in written}
    assert written_names == {"wam_2013.csv", "wam_2013_snap.npz"}

    csv_path = tmp_path / "wam_2013.csv"
    df = pd.read_csv(csv_path)
    # merged target: union of altitudes (250, 400) x union of timestamps (0,1,2) = 6 rows
    assert len(df) == 6

    with np.load(tmp_path / "wam_2013_snap.npz") as npz:
        assert npz["density"].shape == (2, 1, 2, 2)  # H=2 (horizon_hours=1 inclusive), A=1


def test_build_skips_gap_affected_target_but_completes_others(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)

    ok = _avg_check("ok", [{"label": "p", "start": "2013-03-15 00:00:00",
                             "end": "2013-03-15 02:00:00", "physics_avg_csv": "ok.csv"}])
    gappy = _avg_check("gappy", [{"label": "p", "start": "2013-03-15 00:00:00",
                                   "end": "2013-03-15 03:00:00", "physics_avg_csv": "gappy.csv"}])
    suite = ValidationSuite(1, 1, [ok, gappy])

    gap_ts = datetime(2013, 3, 15, 2)  # only referenced by "gappy" (its own end excludes hour 2 for "ok")
    source = _GapSource([gap_ts])

    with pytest.raises(ValueError, match="gappy.csv"):
        build(suite, tmp_path, source)

    assert (tmp_path / "ok.csv").is_file()  # unaffected target still completed
    assert not (tmp_path / "gappy.csv").is_file()  # never a partial file


def test_build_only_check_ids_filters_suite(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)

    keep = _avg_check("keep", [{"label": "p", "start": "2013-03-15 00:00:00",
                                 "end": "2013-03-15 01:00:00", "physics_avg_csv": "keep.csv"}])
    skip = _avg_check("skip", [{"label": "p", "start": "2013-03-15 00:00:00",
                                 "end": "2013-03-15 01:00:00", "physics_avg_csv": "skip.csv"}])
    suite = ValidationSuite(1, 1, [keep, skip])

    written = build(suite, tmp_path, _FakeSource(), only_check_ids=["keep"])
    assert {p.name for p in written} == {"keep.csv"}


def test_build_calls_progress_once_per_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)

    check = _avg_check("c1", [{"label": "p", "start": "2013-03-15 00:00:00",
                                "end": "2013-03-15 03:00:00", "physics_avg_csv": "c1.csv"}])
    suite = ValidationSuite(1, 1, [check])

    calls = []
    build(suite, tmp_path, _FakeSource(), progress=lambda i, total, ts: calls.append((i, total, ts)))

    assert [c[:2] for c in calls] == [(0, 3), (1, 3), (2, 3)]
    assert [c[2] for c in calls] == [datetime(2013, 3, 15, 0), datetime(2013, 3, 15, 1), datetime(2013, 3, 15, 2)]


def test_build_no_matching_checks_returns_empty(tmp_path):
    suite = ValidationSuite(1, 1, [{"id": "x", "kind": "some_other_kind"}])
    assert build(suite, tmp_path, _FakeSource()) == []


def _write_satellite_csv(path, rows):
    lines = ["datetime,lst,lat,lon,alt_km,density"]
    for dt, lst, lat, lon, alt_km, density in rows:
        lines.append(f"{dt},{lst},{lat},{lon},{alt_km},{density}")
    path.write_text("\n".join(lines) + "\n")


def _sat_check(check_id, periods):
    return {"id": check_id, "kind": "satellite_orbit_density", "periods": periods}


def _track_period(label, start, end, sat_csv, phys_csv="phys.csv"):
    return {"label": label, "start": start, "end": end,
            "satellite_track_csv": sat_csv, "physics_model_track_csv": phys_csv}


def test_collect_track_csv_targets_missing_satellite_csv_raises(tmp_path):
    checks = [_sat_check("sat", [_track_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "missing.csv")])]
    with pytest.raises(ValueError, match="satellite ingestion"):
        collect_track_csv_targets(checks, tmp_path)


def test_collect_track_csv_targets_missing_lon_column_raises(tmp_path):
    (tmp_path / "sat.csv").write_text(
        "datetime,lst,lat,alt_km,density\n2023-01-01 00:00:00,12.0,0.0,500.0,1e-12\n"
    )
    checks = [_sat_check("sat", [_track_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "sat.csv")])]
    with pytest.raises(ValueError, match="lon"):
        collect_track_csv_targets(checks, tmp_path)


def test_collect_track_csv_targets_boundary_ceil_hour_included(tmp_path):
    _write_satellite_csv(tmp_path / "sat.csv", [("2023-01-01 23:50:00", 12.0, 0.0, 45.0, 500.0, 0.0)])
    checks = [_sat_check("sat", [_track_period("p", "2023-01-01 23:00:00", "2023-01-01 23:55:00", "sat.csv")])]

    targets = collect_track_csv_targets(checks, tmp_path)
    timestamps = targets["phys.csv"].timestamps
    assert datetime(2023, 1, 1, 23) in timestamps
    assert datetime(2023, 1, 2, 0) in timestamps  # ceil hour, after period.end -- still included


def test_collect_track_csv_targets_conflicting_source_raises(tmp_path):
    _write_satellite_csv(tmp_path / "sat_a.csv", [("2023-01-01 00:00:00", 12.0, 0.0, 45.0, 500.0, 0.0)])
    _write_satellite_csv(tmp_path / "sat_b.csv", [("2023-01-01 00:00:00", 12.0, 0.0, 45.0, 500.0, 0.0)])
    checks = [_sat_check("sat", [
        _track_period("p1", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "sat_a.csv"),
        _track_period("p2", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "sat_b.csv"),
    ])]
    with pytest.raises(ValueError, match="conflicting"):
        collect_track_csv_targets(checks, tmp_path)


def _fake_read_wam_frame(dt, *, variable_names=None):
    """Spatially uniform density that depends only on the hour -- makes the spatial half of the
    interpolation trivially exact regardless of the satellite's lon/lat/alt, isolating the temporal
    weighting for hand computation."""
    lon_values = np.array([0.0, 90.0, 180.0, 270.0])
    lat_values = np.array([-50.0, 0.0, 50.0])
    alt_values = np.array([100.0, 900.0])
    density = np.full((2, 4, 3), float(dt.hour) * 10.0 + 10.0)
    return [WamFrame(time=pd.Timestamp(dt), lon_values=lon_values, lat_values=lat_values,
                      alt_values=alt_values, density=density)]


def test_build_dispatches_scalar_and_track_targets_sharing_an_hour(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_frame", _fake_read_wam_frame)

    _write_satellite_csv(tmp_path / "sat.csv", [("2023-01-01 00:00:00", 12.0, 0.0, 45.0, 500.0, 0.0)])

    avg_check = _avg_check("avg", [{"label": "p", "start": "2023-01-01 00:00:00", "end": "2023-01-01 01:00:00",
                                     "physics_avg_csv": "avg.csv"}])
    sat_check = _sat_check("sat", [_track_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "sat.csv",
                                                  phys_csv="wam_phys.csv")])

    suite = ValidationSuite(1, 1, [avg_check, sat_check])
    source = _FakeSource()
    written = build(suite, tmp_path, source, suite_dir=tmp_path)

    assert source.fetches.count(datetime(2023, 1, 1, 0)) == 1
    assert {p.name for p in written} == {"avg.csv", "wam_phys.csv"}


def test_build_track_csv_temporal_interpolation_hand_computed(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_frame", _fake_read_wam_frame)

    _write_satellite_csv(tmp_path / "sat.csv", [
        ("2023-01-01 00:00:00", 12.0, 0.0, 45.0, 500.0, 0.0),  # weight 0 -> hour0 density (10.0)
        ("2023-01-01 00:30:00", 12.0, 0.0, 45.0, 500.0, 0.0),  # weight 0.5 -> (10+20)/2 = 15.0
    ])
    sat_check = _sat_check("sat", [_track_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "sat.csv",
                                                  phys_csv="wam_phys.csv")])
    suite = ValidationSuite(1, 1, [sat_check])
    source = _FakeSource()
    written = build(suite, tmp_path, source, suite_dir=tmp_path)

    df = pd.read_csv(written[0])
    assert df["density"].iloc[0] == pytest.approx(10.0)
    assert df["density"].iloc[1] == pytest.approx(15.0)


def test_build_satellite_orbit_density_without_suite_dir_raises(tmp_path):
    suite = ValidationSuite(1, 1, [_sat_check("sat", [
        _track_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "sat.csv"),
    ])])
    with pytest.raises(ValueError, match="suite_dir"):
        build(suite, tmp_path, _FakeSource())
