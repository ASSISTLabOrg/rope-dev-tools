"""satellite_source: SwarmDissSatelliteSource (against a fake urlopen), LocalMirrorSatelliteSource, config loader."""

from __future__ import annotations

import json
import urllib.error
from datetime import date

import pytest

from rope_dev_tools.validation.satellite_source import (
    LocalMirrorSatelliteSource,
    SatelliteSourceConfigError,
    SatelliteSourceGapError,
    SwarmDissSatelliteSource,
    _default_mission_for_year,
    load_satellite_source_config,
)


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeUrlopen:
    def __init__(self, *, list_results=None, download_content=b"fake cdf bytes", raise_404_for_list=False):
        self.list_results = list_results if list_results is not None else []
        self.download_content = download_content
        self.raise_404_for_list = raise_404_for_list
        self.calls = []

    def __call__(self, req, timeout=None, context=None):
        url = req.full_url if hasattr(req, "full_url") else req
        self.calls.append(url)
        if "do=list" in url:
            if self.raise_404_for_list:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return _FakeResponse(json.dumps({"success": True, "results": self.list_results}).encode())
        if "do=download" in url:
            return _FakeResponse(self.download_content)
        raise AssertionError(f"unexpected URL: {url}")


def _entry(name, is_dir=False):
    return {"name": name, "path": f"some/dir/{name}", "is_dir": is_dir, "size": 100}


def test_default_mission_for_year_cutover():
    assert _default_mission_for_year(2017) == "GRACE"
    assert _default_mission_for_year(2018) == "GRACE-FO"


def test_swarm_diss_fetch_day_success(tmp_path, monkeypatch):
    fake = _FakeUrlopen(list_results=[_entry("GF_OPER_DNS1ACC_2__20230101T000000_20230101T235959_0003.cdf")])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    source = SwarmDissSatelliteSource({})
    local_path = source.fetch_day(date(2023, 1, 1), tmp_path / "scratch")

    assert local_path.is_file()
    assert local_path.read_bytes() == b"fake cdf bytes"
    assert any("do=list" in c for c in fake.calls)
    assert any("do=download" in c for c in fake.calls)
    assert any("GRACE-FO" in c for c in fake.calls)  # default mission for 2023


def test_swarm_diss_fetch_day_uses_year_override(tmp_path, monkeypatch):
    fake = _FakeUrlopen(list_results=[_entry("GR_OPER_DNS1ACC_2__20130101T000000_20130101T235959_0001.cdf")])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    source = SwarmDissSatelliteSource({2013: {"mission": "GRACE"}})
    source.fetch_day(date(2013, 1, 1), tmp_path / "scratch")

    assert any("GRACE" in c and "GRACE-FO" not in c for c in fake.calls)


def test_swarm_diss_fetch_day_404_raises_config_error(tmp_path, monkeypatch):
    fake = _FakeUrlopen(raise_404_for_list=True)
    monkeypatch.setattr("urllib.request.urlopen", fake)

    source = SwarmDissSatelliteSource({2023: {"satellite": "Sat_2"}})
    with pytest.raises(SatelliteSourceConfigError):
        source.fetch_day(date(2023, 1, 1), tmp_path / "scratch")


def test_swarm_diss_fetch_day_no_match_raises_gap_error(tmp_path, monkeypatch):
    fake = _FakeUrlopen(list_results=[_entry("GR_OPER_DNS1ACC_2__20130101T000000_20130101T235959_0001.cdf")])
    monkeypatch.setattr("urllib.request.urlopen", fake)

    source = SwarmDissSatelliteSource({2013: {"mission": "GRACE"}})
    with pytest.raises(SatelliteSourceGapError):
        source.fetch_day(date(2013, 3, 15), tmp_path / "scratch")


def test_local_mirror_fetch_day_finds_matching_file(tmp_path):
    year_dir = tmp_path / "2013"
    year_dir.mkdir()
    matching = year_dir / "GR_OPER_DNS1ACC_2__20130315T000000_20130315T235959_0001.cdf"
    matching.write_bytes(b"data")

    source = LocalMirrorSatelliteSource({2013: {"dir": str(year_dir), "mission": "GRACE"}})
    resolved = source.fetch_day(date(2013, 3, 15), tmp_path / "scratch")
    assert resolved == matching


def test_local_mirror_fetch_day_missing_year_raises(tmp_path):
    source = LocalMirrorSatelliteSource({})
    with pytest.raises(SatelliteSourceConfigError):
        source.fetch_day(date(2013, 3, 15), tmp_path / "scratch")


def test_local_mirror_fetch_day_no_match_raises_gap_error(tmp_path):
    year_dir = tmp_path / "2013"
    year_dir.mkdir()
    source = LocalMirrorSatelliteSource({2013: {"dir": str(year_dir), "mission": "GRACE"}})
    with pytest.raises(SatelliteSourceGapError):
        source.fetch_day(date(2013, 3, 15), tmp_path / "scratch")


def test_local_mirror_release_is_noop(tmp_path):
    year_dir = tmp_path / "2013"
    year_dir.mkdir()
    matching = year_dir / "GR_OPER_DNS1ACC_2__20130315T000000_20130315T235959_0001.cdf"
    matching.write_bytes(b"data")

    source = LocalMirrorSatelliteSource({2013: {"dir": str(year_dir), "mission": "GRACE"}})
    resolved = source.fetch_day(date(2013, 3, 15), tmp_path / "scratch")
    source.release(resolved)
    assert resolved.is_file()


def test_load_satellite_source_config_round_trip(tmp_path):
    config_path = tmp_path / "satellite_sources.json"
    config_path.write_text(json.dumps({
        "default_satellite": "Sat_2",
        "remote": {"years": {"2013": {"mission": "GRACE"}}},
        "offline": {"years": {"2013": {"dir": "mirror/2013", "mission": "GRACE"}}},
    }))

    config = load_satellite_source_config(config_path)
    assert config["default_satellite"] == "Sat_2"
    assert config["remote"][2013] == {"mission": "GRACE"}
    assert config["offline"][2013]["dir"] == str(tmp_path / "mirror" / "2013")


def test_load_satellite_source_config_non_integer_year_raises(tmp_path):
    config_path = tmp_path / "satellite_sources.json"
    config_path.write_text(json.dumps({"remote": {"years": {"not-a-year": {"mission": "GRACE"}}}}))
    with pytest.raises(SatelliteSourceConfigError):
        load_satellite_source_config(config_path)
