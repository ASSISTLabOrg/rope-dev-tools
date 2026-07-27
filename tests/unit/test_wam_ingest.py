"""wam_ingest: target collection (dedup/union across checks) and the streaming fetch-once build loop."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rope_dev_tools.validation.schema_types import ValidationSuite
from rope_dev_tools.validation.wam_convert import WamTimestep
from rope_dev_tools.validation.wam_ingest import (
    _hourly_timestamps,
    _hourly_timestamps_inclusive,
    build,
    collect_avg_density_targets,
    collect_hourly_npz_targets,
)
from rope_dev_tools.validation.wam_source import WamRawDataSource


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


def test_collect_hourly_npz_targets_unions_altitudes_for_shared_filename():
    check_a = {"id": "a", "kind": "lonlat_snapshot_series", "start": "2023-01-01 00:00:00",
               "horizon_hours": 2, "physics_model_hourly_npz": "shared.npz", "altitudes_km": [300.0]}
    check_b = {"id": "b", "kind": "lonlat_snapshot_series", "start": "2023-01-01 00:00:00",
               "horizon_hours": 2, "physics_model_hourly_npz": "shared.npz", "altitudes_km": [400.0]}
    targets = collect_hourly_npz_targets([check_a, check_b])
    assert set(targets) == {"shared.npz"}
    assert targets["shared.npz"].altitudes_km == {300.0, 400.0}
    assert len(targets["shared.npz"].timestamps) == 3


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


def _fake_read_wam_timesteps(dt, *, altitudes_km, variable_names=None):
    grid = np.zeros((2, 2))
    return [WamTimestep(
        time=pd.Timestamp(dt),
        grid_mean_density={alt: float(dt.hour) + alt for alt in altitudes_km},
        lst_lat_density={alt: grid for alt in altitudes_km},
        n_lst=2, n_lat=2, lat_min_deg=-10.0, lat_max_deg=10.0,
    )]


def test_build_fetches_shared_timestamp_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)

    storm = _avg_check("storms", [{"label": "March 2013", "start": "2013-03-15 00:00:00",
                                    "end": "2013-03-15 02:00:00", "physics_avg_csv": "wam_2013.csv"}],
                        altitudes_km=(250.0,))
    yearly = _avg_check("yearly", [{"label": "2013 span", "start": "2013-03-15 00:00:00",
                                     "end": "2013-03-15 03:00:00", "physics_avg_csv": "wam_2013.csv"}],
                         altitudes_km=(400.0,))
    lonlat = {"id": "snap", "kind": "lonlat_snapshot_series", "start": "2013-03-15 00:00:00",
              "horizon_hours": 1, "physics_model_hourly_npz": "wam_2013_snap.npz", "altitudes_km": [300.0]}

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


def test_build_only_check_ids_filters_suite(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.wam_ingest.read_wam_timesteps", _fake_read_wam_timesteps)

    keep = _avg_check("keep", [{"label": "p", "start": "2013-03-15 00:00:00",
                                 "end": "2013-03-15 01:00:00", "physics_avg_csv": "keep.csv"}])
    skip = _avg_check("skip", [{"label": "p", "start": "2013-03-15 00:00:00",
                                 "end": "2013-03-15 01:00:00", "physics_avg_csv": "skip.csv"}])
    suite = ValidationSuite(1, 1, [keep, skip])

    written = build(suite, tmp_path, _FakeSource(), only_check_ids=["keep"])
    assert {p.name for p in written} == {"keep.csv"}


def test_build_no_matching_checks_returns_empty(tmp_path):
    suite = ValidationSuite(1, 1, [{"id": "x", "kind": "some_other_kind"}])
    assert build(suite, tmp_path, _FakeSource()) == []
