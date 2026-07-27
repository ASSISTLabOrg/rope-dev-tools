"""wam_source — resolves one UTC timestamp to a local raw WAM .nc file, via S3 or a local mirror."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from rope_dev_tools.validation.time_utils import resolve_path

# One file per hour, flat under the year's prefix; only on-the-hour timestamps are ever requested.
DEFAULT_FILENAME_PATTERN = "wam_fixed_height.wam.%Y%m%d_%H0000.nc"


class WamSourceConfigError(ValueError):
    pass


class WamRawDataSource(ABC):
    @abstractmethod
    def fetch_timestep(self, dt: datetime, scratch_dir: Path) -> Path:
        """Makes the raw .nc file covering this exact UTC hour available locally, returns its path."""
        raise NotImplementedError

    def release(self, path: Path) -> None:
        """Called once nothing else needs this timestep's data. Default: delete the local file."""
        Path(path).unlink(missing_ok=True)


class LocalMirrorWamSource(WamRawDataSource):
    """Reads from an already-local mirrored copy of the raw WAM archive (the offline-dev case)."""

    def __init__(self, year_config: dict, *, default_filename_pattern: str = DEFAULT_FILENAME_PATTERN):
        self._year_config = year_config
        self._default_pattern = default_filename_pattern

    def fetch_timestep(self, dt: datetime, scratch_dir: Path) -> Path:
        try:
            year_cfg = self._year_config[dt.year]
        except KeyError:
            raise WamSourceConfigError(f"no local-mirror location configured for year {dt.year}") from None

        pattern = year_cfg.get("filename_pattern", self._default_pattern)
        local_path = Path(year_cfg["dir"]) / dt.strftime(pattern)
        if not local_path.is_file():
            raise WamSourceConfigError(f"no local WAM file found at {local_path} for timestep {dt}")
        return local_path

    def release(self, path: Path) -> None:
        return None  # dev's own mirrored archive — never delete it


class S3WamSource(WamRawDataSource):
    """Streams raw WAM data from S3, one timestep file at a time."""

    def __init__(self, year_config: dict, *, bucket: str, default_filename_pattern: str = DEFAULT_FILENAME_PATTERN):
        self._year_config = year_config
        self._bucket = bucket
        self._default_pattern = default_filename_pattern
        self._client = None

    def _s3_client(self):
        if self._client is None:
            import boto3  # lazy import — only S3WamSource needs boto3/network

            session = boto3.Session()
            if session.get_credentials() is None:
                from botocore import UNSIGNED
                from botocore.config import Config

                self._client = session.client("s3", config=Config(signature_version=UNSIGNED))
            else:
                self._client = session.client("s3")
        return self._client

    def fetch_timestep(self, dt: datetime, scratch_dir: Path) -> Path:
        try:
            year_cfg = self._year_config[dt.year]
        except KeyError:
            raise WamSourceConfigError(f"no S3 location configured for year {dt.year}") from None

        pattern = year_cfg.get("filename_pattern", self._default_pattern)
        relative_key = dt.strftime(pattern)
        key = year_cfg["prefix"].rstrip("/") + "/" + relative_key

        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        local_path = scratch_dir / Path(relative_key).name
        self._s3_client().download_file(self._bucket, key, str(local_path))
        return local_path


def load_wam_source_config(path) -> dict:
    """Loads rope-data/validation/wam_sources.json — see docs/ingesting-wam-data.md for the format."""
    path = Path(path)
    raw = json.loads(path.read_text())
    out = {"default_filename_pattern": raw.get("default_filename_pattern", DEFAULT_FILENAME_PATTERN)}

    for source_kind in ("s3", "offline"):
        block = raw.get(source_kind)
        if block is None:
            continue
        years = {}
        for key, entry in block.get("years", {}).items():
            try:
                year = int(key)
            except ValueError:
                raise WamSourceConfigError(f"{path}: {source_kind}.years key {key!r} is not an integer") from None
            entry = dict(entry)
            if source_kind == "offline" and "dir" in entry:
                entry["dir"] = str(resolve_path(path.parent, entry["dir"]))
            years[year] = entry
        out[source_kind] = {"bucket": block.get("bucket"), "years": years}
    return out
