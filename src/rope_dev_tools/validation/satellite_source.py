"""satellite_source — resolves one calendar day to a local raw satellite CDF file, via swarm-diss or a local mirror."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from rope_dev_tools.validation.time_utils import resolve_path

DEFAULT_SATELLITE = "Sat_1"
_MISSION_CUTOVER_YEAR = 2018  # < this -> GRACE, >= this -> GRACE-FO; override-able per year
SWARM_DISS_BASE_URL = "https://swarm-diss.eo.esa.int/"


def _urlopen(url: str, *, timeout: float):
    """Falls back to no-verify SSL on a local cert-verification failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "rope-dev-tools/1.0"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        if "SSL" not in str(e) and "certificate" not in str(e).lower():
            raise
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def _default_mission_for_year(year: int) -> str:
    return "GRACE" if year < _MISSION_CUTOVER_YEAR else "GRACE-FO"


class SatelliteSourceConfigError(ValueError):
    pass


class SatelliteSourceGapError(ValueError):
    """No file exists upstream for this exact day — expected/collectible, not a bug."""

    def __init__(self, mission: str, satellite: str, day: date):
        self.mission, self.satellite, self.day = mission, satellite, day
        super().__init__(f"no {mission} {satellite} density file found for {day}")


def _mission_prefix(mission: str) -> str:
    if mission == "GRACE":
        return "GR"
    if mission == "GRACE-FO":
        return "GF"
    raise SatelliteSourceConfigError(f"unknown mission {mission!r}; expected 'GRACE' or 'GRACE-FO'")


class SatelliteRawDataSource(ABC):
    @abstractmethod
    def fetch_day(self, day: date, scratch_dir: Path) -> Path:
        """Makes the raw CDF file covering this calendar day available locally, returns its path."""
        raise NotImplementedError

    def release(self, path: Path) -> None:
        Path(path).unlink(missing_ok=True)


class LocalMirrorSatelliteSource(SatelliteRawDataSource):
    """Reads from an already-local mirrored copy of the raw satellite archive."""

    def __init__(self, year_config: dict, *, default_satellite: str = DEFAULT_SATELLITE):
        self._year_config = year_config
        self._default_satellite = default_satellite

    def fetch_day(self, day: date, scratch_dir: Path) -> Path:
        try:
            year_cfg = self._year_config[day.year]
        except KeyError:
            raise SatelliteSourceConfigError(f"no local-mirror location configured for year {day.year}") from None

        mission = year_cfg.get("mission", _default_mission_for_year(day.year))
        satellite = year_cfg.get("satellite", self._default_satellite)
        prefix = _mission_prefix(mission)
        matches = sorted(Path(year_cfg["dir"]).glob(f"{prefix}_*{day:%Y%m%d}T000000*.cdf"))
        if not matches:
            raise SatelliteSourceGapError(mission, satellite, day)
        return matches[0]

    def release(self, path: Path) -> None:
        return None  # dev's own mirrored archive — never delete it


class SwarmDissSatelliteSource(SatelliteRawDataSource):
    """Streams raw satellite density data from the ESA swarm-diss public server, one day at a time."""

    def __init__(self, year_config: dict, *, default_satellite: str = DEFAULT_SATELLITE,
                 base_url: str = SWARM_DISS_BASE_URL):
        self._year_config = year_config
        self._default_satellite = default_satellite
        self._base_url = base_url

    def _list(self, dir_path: str) -> dict:
        url = self._base_url + "?" + urllib.parse.urlencode(
            {"do": "list", "maxfiles": 500, "pos": 0, "file": dir_path}
        )
        try:
            with _urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise SatelliteSourceConfigError(
                    f"{dir_path}: not found upstream — check mission/satellite/year in satellite_sources.json"
                ) from None
            raise
        if not payload.get("success"):
            raise SatelliteSourceConfigError(f"{dir_path}: listing did not succeed: {payload}")
        return payload

    def fetch_day(self, day: date, scratch_dir: Path) -> Path:
        try:
            year_cfg = self._year_config[day.year]
        except KeyError:
            year_cfg = {}

        mission = year_cfg.get("mission", _default_mission_for_year(day.year))
        satellite = year_cfg.get("satellite", self._default_satellite)
        dir_path = f"swarm/Multimission/{mission}/DNS/{satellite}/{day.year}"

        payload = self._list(dir_path)
        needle = f"{day:%Y%m%d}T000000"
        matches = [r for r in payload["results"] if not r["is_dir"] and needle in r["name"]]
        if not matches:
            raise SatelliteSourceGapError(mission, satellite, day)

        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        remote_path = matches[0]["path"]
        local_path = scratch_dir / Path(remote_path).name

        download_url = self._base_url + "?" + urllib.parse.urlencode({"do": "download", "file": remote_path})
        with _urlopen(download_url, timeout=60) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
        return local_path


def load_satellite_source_config(path) -> dict:
    """Loads rope-data/validation/satellite_sources.json — see docs/ingesting-satellite-data.md."""
    path = Path(path)
    raw = json.loads(path.read_text())
    out = {"default_satellite": raw.get("default_satellite", DEFAULT_SATELLITE)}

    for source_kind in ("remote", "offline"):
        block = raw.get(source_kind)
        if block is None:
            continue
        years = {}
        for key, entry in block.get("years", {}).items():
            try:
                year = int(key)
            except ValueError:
                raise SatelliteSourceConfigError(f"{path}: {source_kind}.years key {key!r} is not an integer") from None
            entry = dict(entry)
            if source_kind == "offline" and "dir" in entry:
                entry["dir"] = str(resolve_path(path.parent, entry["dir"]))
            years[year] = entry
        out[source_kind] = years
    return out
