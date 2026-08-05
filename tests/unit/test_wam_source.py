"""wam_source: LocalMirrorWamSource, S3WamSource (against a fake boto3 client), and the config loader."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from botocore.exceptions import ClientError

from rope_dev_tools.validation.wam_source import (
    DEFAULT_FILENAME_PATTERN,
    LocalMirrorWamSource,
    S3WamSource,
    WamSourceConfigError,
    WamSourceGapError,
    load_wam_source_config,
)


def _touch_nc(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real nc file, just needs to exist")


def test_local_mirror_fetch_timestep_resolves_pattern(tmp_path):
    nc_path = tmp_path / "wam_fixed_height.wam.20240511_030000.nc"
    _touch_nc(nc_path)

    source = LocalMirrorWamSource({2024: {"dir": str(tmp_path)}})
    resolved = source.fetch_timestep(datetime(2024, 5, 11, 3), tmp_path / "scratch")
    assert resolved == nc_path


def test_local_mirror_missing_year_raises(tmp_path):
    source = LocalMirrorWamSource({2024: {"dir": str(tmp_path)}})
    with pytest.raises(WamSourceConfigError, match="2025"):
        source.fetch_timestep(datetime(2025, 1, 1), tmp_path / "scratch")


def test_local_mirror_missing_file_raises(tmp_path):
    source = LocalMirrorWamSource({2024: {"dir": str(tmp_path)}})
    with pytest.raises(WamSourceConfigError):
        source.fetch_timestep(datetime(2024, 5, 11, 3), tmp_path / "scratch")


def test_local_mirror_release_is_noop(tmp_path):
    nc_path = tmp_path / "wam_fixed_height.wam.20240511_030000.nc"
    _touch_nc(nc_path)
    source = LocalMirrorWamSource({2024: {"dir": str(tmp_path)}})
    resolved = source.fetch_timestep(datetime(2024, 5, 11, 3), tmp_path / "scratch")
    source.release(resolved)
    assert resolved.is_file()


def test_local_mirror_per_year_filename_pattern_override(tmp_path):
    nc_path = tmp_path / "custom" / "2024-05-11-03.nc"
    _touch_nc(nc_path)
    source = LocalMirrorWamSource({2024: {"dir": str(tmp_path), "filename_pattern": "custom/%Y-%m-%d-%H.nc"}})
    resolved = source.fetch_timestep(datetime(2024, 5, 11, 3), tmp_path / "scratch")
    assert resolved == nc_path


class _FakeS3Client:
    """existing_keys=None means every key exists (head_object always succeeds); otherwise only those keys do."""

    def __init__(self, existing_keys=None):
        self.download_calls = []
        self.head_calls = []
        self._existing_keys = None if existing_keys is None else set(existing_keys)

    def head_object(self, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        if self._existing_keys is not None and Key not in self._existing_keys:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    def download_file(self, bucket, key, local_path):
        self.download_calls.append((bucket, key, local_path))
        with open(local_path, "wb") as f:
            f.write(b"fake downloaded content")


class _FakeSession:
    def __init__(self, client):
        self._client = client

    def get_credentials(self):
        return None

    def client(self, service, **kwargs):
        assert service == "s3"
        return self._client


def test_s3_source_fetch_timestep_builds_key_and_downloads(tmp_path, monkeypatch):
    fake_client = _FakeS3Client()
    monkeypatch.setattr("boto3.Session", lambda: _FakeSession(fake_client))

    source = S3WamSource({2024: {"prefix": "some/prefix/for/2024/"}}, bucket="my-bucket")
    local_path = source.fetch_timestep(datetime(2024, 5, 11, 3), tmp_path / "scratch")

    assert local_path.is_file()
    assert fake_client.download_calls == [
        ("my-bucket", "some/prefix/for/2024/wam_fixed_height.wam.20240511_030000.nc", str(local_path)),
    ]


def test_s3_source_missing_year_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("boto3.Session", lambda: _FakeSession(_FakeS3Client()))
    # 2025's neighbors (2024, 2026) aren't configured either -- nothing to even try.
    source = S3WamSource({2000: {"prefix": "p/"}}, bucket="my-bucket")
    with pytest.raises(WamSourceConfigError, match="2025"):
        source.fetch_timestep(datetime(2025, 1, 1), tmp_path / "scratch")


def test_s3_source_falls_back_to_prior_year_when_absent_from_primary(tmp_path, monkeypatch):
    # Real-world case: a year's research run overlaps into the next year, so the Jan-1 00:00 file
    # for year N actually lives under year N-1's prefix, not year N's own.
    key_2008 = "p/2008/wam_fixed_height.wam.20090101_000000.nc"
    fake_client = _FakeS3Client(existing_keys={key_2008})
    monkeypatch.setattr("boto3.Session", lambda: _FakeSession(fake_client))

    source = S3WamSource({2008: {"prefix": "p/2008/"}, 2009: {"prefix": "p/2009/"}}, bucket="my-bucket")
    local_path = source.fetch_timestep(datetime(2009, 1, 1, 0), tmp_path / "scratch")

    assert local_path.is_file()
    assert fake_client.head_calls == [
        ("my-bucket", "p/2009/wam_fixed_height.wam.20090101_000000.nc"),
        ("my-bucket", key_2008),
    ]
    assert fake_client.download_calls == [("my-bucket", key_2008, str(local_path))]


def test_s3_source_falls_back_to_next_year_when_absent_from_primary(tmp_path, monkeypatch):
    key_2011 = "p/2011/wam_fixed_height.wam.20101231_230000.nc"
    fake_client = _FakeS3Client(existing_keys={key_2011})
    monkeypatch.setattr("boto3.Session", lambda: _FakeSession(fake_client))

    source = S3WamSource({2010: {"prefix": "p/2010/"}, 2011: {"prefix": "p/2011/"}}, bucket="my-bucket")
    local_path = source.fetch_timestep(datetime(2010, 12, 31, 23), tmp_path / "scratch")

    assert local_path.is_file()
    assert fake_client.download_calls == [("my-bucket", key_2011, str(local_path))]


def test_s3_source_raises_gap_error_when_absent_from_every_configured_year(tmp_path, monkeypatch):
    fake_client = _FakeS3Client(existing_keys=set())  # nothing exists anywhere
    monkeypatch.setattr("boto3.Session", lambda: _FakeSession(fake_client))

    source = S3WamSource({2008: {"prefix": "p/2008/"}, 2009: {"prefix": "p/2009/"}}, bucket="my-bucket")
    with pytest.raises(WamSourceGapError) as exc_info:
        source.fetch_timestep(datetime(2009, 1, 1, 0), tmp_path / "scratch")
    assert exc_info.value.years_tried == [2009, 2008]


def test_s3_source_release_deletes_local_file(tmp_path, monkeypatch):
    fake_client = _FakeS3Client()
    monkeypatch.setattr("boto3.Session", lambda: _FakeSession(fake_client))
    source = S3WamSource({2024: {"prefix": "p/"}}, bucket="my-bucket")
    local_path = source.fetch_timestep(datetime(2024, 5, 11, 3), tmp_path / "scratch")
    assert local_path.is_file()
    source.release(local_path)
    assert not local_path.is_file()


def test_load_wam_source_config_round_trip(tmp_path):
    config_path = tmp_path / "wam_sources.json"
    config_path.write_text(json.dumps({
        "default_filename_pattern": "custom_%Y.nc",
        "s3": {"bucket": "my-bucket", "years": {"2024": {"prefix": "p/2024/"}}},
        "offline": {"years": {"2024": {"dir": "mirror/2024"}}},
    }))

    config = load_wam_source_config(config_path)
    assert config["default_filename_pattern"] == "custom_%Y.nc"
    assert config["s3"]["bucket"] == "my-bucket"
    assert config["s3"]["years"][2024] == {"prefix": "p/2024/"}
    # offline dir is resolved relative to the config file's own directory
    assert config["offline"]["years"][2024]["dir"] == str(tmp_path / "mirror" / "2024")


def test_load_wam_source_config_missing_blocks_are_absent(tmp_path):
    config_path = tmp_path / "wam_sources.json"
    config_path.write_text(json.dumps({"s3": {"bucket": "b", "years": {}}}))
    config = load_wam_source_config(config_path)
    assert "offline" not in config
    assert config["default_filename_pattern"] == DEFAULT_FILENAME_PATTERN


def test_load_wam_source_config_non_integer_year_raises(tmp_path):
    config_path = tmp_path / "wam_sources.json"
    config_path.write_text(json.dumps({"s3": {"bucket": "b", "years": {"twenty-twenty-four": {"prefix": "p/"}}}}))
    with pytest.raises(WamSourceConfigError):
        load_wam_source_config(config_path)
