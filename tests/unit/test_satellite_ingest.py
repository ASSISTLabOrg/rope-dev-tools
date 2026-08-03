"""satellite_ingest: target dedup and the streaming fetch-once/gap-collection build loop."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rope_dev_tools.validation.satellite_ingest import build, collect_satellite_track_targets
from rope_dev_tools.validation.satellite_source import SatelliteRawDataSource, SatelliteSourceGapError
from rope_dev_tools.validation.schema_types import ValidationSuite


def _check(check_id, periods):
    return {"id": check_id, "kind": "satellite_orbit_density", "periods": periods}


def _period(label, start, end, sat_csv, phys_csv="phys.csv"):
    return {"label": label, "start": start, "end": end,
            "satellite_track_csv": sat_csv, "physics_model_track_csv": phys_csv}


def test_collect_satellite_track_targets_unions_dates_across_checks():
    check_a = _check("a", [_period("p", "2023-01-01 00:00:00", "2023-01-03 00:00:00", "shared.csv")])
    check_b = _check("b", [_period("p", "2023-01-02 00:00:00", "2023-01-04 00:00:00", "shared.csv")])

    targets = collect_satellite_track_targets([check_a, check_b])
    assert set(targets) == {"shared.csv"}
    assert targets["shared.csv"].dates == {date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)}


class _FakeSource(SatelliteRawDataSource):
    def __init__(self, gap_dates=frozenset()):
        self.gap_dates = frozenset(gap_dates)
        self.fetches = []
        self.releases = []

    def fetch_day(self, day, scratch_dir):
        self.fetches.append(day)
        if day in self.gap_dates:
            raise SatelliteSourceGapError("GRACE", "Sat_1", day)
        return day  # sentinel "path" -- read_satellite_day is monkeypatched to accept it

    def release(self, path):
        self.releases.append(path)


def _fake_read_satellite_day(day, cadence_seconds=600):
    return pd.DataFrame({
        "datetime": [pd.Timestamp(day)], "lst": [12.0], "lat": [0.0], "lon": [0.0],
        "alt_km": [500.0], "density": [float(day.day)],
    })


def test_build_fetches_shared_day_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.satellite_ingest.read_satellite_day", _fake_read_satellite_day)

    check_a = _check("a", [_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "a.csv")])
    check_b = _check("b", [_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "b.csv")])
    suite = ValidationSuite(1, 1, [check_a, check_b])

    source = _FakeSource()
    written = build(suite, tmp_path, source)

    assert source.fetches.count(date(2023, 1, 1)) == 1
    assert len(source.fetches) == len(source.releases)
    assert {p.name for p in written} == {"a.csv", "b.csv"}


def test_build_gap_collected_other_targets_still_written(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.satellite_ingest.read_satellite_day", _fake_read_satellite_day)

    good = _check("good", [_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "good.csv")])
    gapped = _check("gapped", [_period("p", "2013-03-15 00:00:00", "2013-03-17 00:00:00", "gapped.csv")])
    suite = ValidationSuite(1, 1, [good, gapped])

    source = _FakeSource(gap_dates={date(2013, 3, 15), date(2013, 3, 16)})
    with pytest.raises(ValueError) as excinfo:
        build(suite, tmp_path, source)

    assert "2013-03-15" in str(excinfo.value)
    assert "2013-03-16" in str(excinfo.value)
    assert "gapped.csv" in str(excinfo.value)
    assert (tmp_path / "good.csv").is_file()
    assert not (tmp_path / "gapped.csv").exists()


def test_build_non_gap_exception_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.satellite_ingest.read_satellite_day", _fake_read_satellite_day)

    class _BrokenSource(SatelliteRawDataSource):
        def fetch_day(self, day, scratch_dir):
            raise RuntimeError("network exploded")

        def release(self, path):
            pass

    suite = ValidationSuite(1, 1, [_check("a", [_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "a.csv")])])
    with pytest.raises(RuntimeError, match="network exploded"):
        build(suite, tmp_path, _BrokenSource())


def test_build_only_check_ids_filters_suite(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.satellite_ingest.read_satellite_day", _fake_read_satellite_day)

    keep = _check("keep", [_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "keep.csv")])
    skip = _check("skip", [_period("p", "2023-01-01 00:00:00", "2023-01-02 00:00:00", "skip.csv")])
    suite = ValidationSuite(1, 1, [keep, skip])

    written = build(suite, tmp_path, _FakeSource(), only_check_ids=["keep"])
    assert {p.name for p in written} == {"keep.csv"}


def test_build_calls_progress_once_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr("rope_dev_tools.validation.satellite_ingest.read_satellite_day", _fake_read_satellite_day)

    check = _check("c1", [_period("p", "2023-01-01 00:00:00", "2023-01-03 00:00:00", "c1.csv")])
    suite = ValidationSuite(1, 1, [check])

    calls = []
    build(suite, tmp_path, _FakeSource(), progress=lambda i, total, d: calls.append((i, total, d)))

    assert [c[:2] for c in calls] == [(0, 2), (1, 2)]
    assert [c[2] for c in calls] == [date(2023, 1, 1), date(2023, 1, 2)]


def test_build_no_matching_checks_returns_empty(tmp_path):
    suite = ValidationSuite(1, 1, [{"id": "x", "kind": "some_other_kind"}])
    assert build(suite, tmp_path, _FakeSource()) == []
