"""Two interchangeable ModelInterface backends: wrapper mode and exported-dir mode."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rope_dev_tools.validation.time_utils import hours_between, parse_time

PACKAGE_ROOT_ENV = "ROPE_PACKAGE_ROOT"
BUILD_DIR_ENV = "ROPE_BUILD_DIR"


class ModelInterface(ABC):
    backend_name: str = "unknown"

    @property
    @abstractmethod
    def grid(self) -> dict:
        """GridSpec-shaped dict ({"n_lst", "n_lat", "n_alt", "lat_min_deg", "lat_max_deg", "alt_min_km", "alt_max_km"}) this model's forecasts are on."""
        raise NotImplementedError

    @abstractmethod
    def forecast(self, start: str, end: str) -> dict:
        """Forecasts [start, end]. Returns {"window_start", "window_end"} — the queryable window, not [start, end] itself."""
        raise NotImplementedError

    @abstractmethod
    def query(self, time: str, lst: float, lat: float, alt_km: float) -> dict:
        """Returns {"density", "uncertainty"} at a single point."""
        raise NotImplementedError

    @abstractmethod
    def query_grid(self, time: str, alt_km: float) -> np.ndarray:
        """Returns an (n_lst, n_lat) density array at the given time/altitude."""
        raise NotImplementedError

    @abstractmethod
    def query_grid_at(self, time: str, alt_km: float, lst_values: np.ndarray, lat_values: np.ndarray) -> np.ndarray:
        """Returns a (len(lst_values), len(lat_values)) density array at the given axis values."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook; a no-op unless a backend owns a process/handle."""
        return None


# Wrapper mode

@dataclass
class WrapperRequest:
    start: str
    end: str


@dataclass
class WrapperResponse:
    times: list             # ISO timestamps, length T
    density: np.ndarray      # (T, grid["n_lst"], grid["n_lat"], grid["n_alt"])
    uncertainty: np.ndarray  # same shape


WrapperFn = Callable[[WrapperRequest], WrapperResponse]


def _lst_index(lst: float, grid: dict) -> int:
    """Nearest LST grid index, wrapping at 24h."""
    n_lst = grid["n_lst"]
    return int(round((lst % 24.0) / 24.0 * n_lst)) % n_lst


def _lat_index(lat: float, grid: dict) -> int:
    """Nearest lat grid index, clamped to the grid's range."""
    lat_min, lat_max, n_lat = grid["lat_min_deg"], grid["lat_max_deg"], grid["n_lat"]
    lat = min(max(lat, lat_min), lat_max)
    frac = (lat - lat_min) / (lat_max - lat_min)
    return int(round(frac * (n_lat - 1)))


def _alt_index(alt_km: float, grid: dict) -> int:
    """Nearest altitude grid index, clamped to the grid's range."""
    alt_min, alt_max, n_alt = grid["alt_min_km"], grid["alt_max_km"], grid["n_alt"]
    alt_km = min(max(alt_km, alt_min), alt_max)
    frac = (alt_km - alt_min) / (alt_max - alt_min)
    return int(round(frac * (n_alt - 1)))


def _nearest_time_index(times: list, time: str) -> int:
    """Index into times of the entry closest to time."""
    target = parse_time(time)
    diffs = [abs((parse_time(t) - target).total_seconds()) for t in times]
    return diffs.index(min(diffs))


class WrapperModelInterface(ModelInterface):
    """Drives a dev-supplied callable directly, using nearest-grid-cell lookup."""

    backend_name = "wrapper"

    def __init__(self, wrapper_fn: WrapperFn, grid: dict):
        self._wrapper_fn = wrapper_fn
        self._grid = grid
        self._response: "WrapperResponse | None" = None

    @property
    def grid(self) -> dict:
        return self._grid

    def forecast(self, start: str, end: str) -> dict:
        self._response = self._wrapper_fn(WrapperRequest(start=start, end=end))
        return {"window_start": self._response.times[0], "window_end": self._response.times[-1]}

    def query(self, time: str, lst: float, lat: float, alt_km: float) -> dict:
        if self._response is None:
            raise RuntimeError("forecast() must be called before query()")
        ti = _nearest_time_index(self._response.times, time)
        li = _lst_index(lst, self._grid)
        ai = _lat_index(lat, self._grid)
        alti = _alt_index(alt_km, self._grid)
        density = float(self._response.density[ti, li, ai, alti])
        uncertainty = float(self._response.uncertainty[ti, li, ai, alti])
        return {"density": density, "uncertainty": uncertainty}

    def query_grid(self, time: str, alt_km: float) -> np.ndarray:
        if self._response is None:
            raise RuntimeError("forecast() must be called before query_grid()")
        ti = _nearest_time_index(self._response.times, time)
        alti = _alt_index(alt_km, self._grid)
        return np.asarray(self._response.density[ti, :, :, alti])

    def query_grid_at(self, time: str, alt_km: float, lst_values: np.ndarray, lat_values: np.ndarray) -> np.ndarray:
        if self._response is None:
            raise RuntimeError("forecast() must be called before query_grid_at()")
        ti = _nearest_time_index(self._response.times, time)
        alti = _alt_index(alt_km, self._grid)
        li = [_lst_index(lst, self._grid) for lst in lst_values]
        ai = [_lat_index(lat, self._grid) for lat in lat_values]
        return np.asarray(self._response.density[ti][np.ix_(li, ai)][:, :, alti])


# Exported-directory mode

class RopePackageNotFoundError(RuntimeError):
    pass


def _discover_package_root(*, require_binary: bool = True) -> Path:
    """require_binary=False when build_dir is given explicitly (only need python/rope.py then)."""
    override = os.environ.get(PACKAGE_ROOT_ENV)
    if override:
        return Path(override)

    candidates = [Path.cwd() / "rope-framework", Path.cwd().parent / "rope-framework"]
    for root in candidates:
        if not (root / "python" / "rope.py").is_file():
            continue
        if not require_binary or (root / "build" / "rope").is_file() or (root / "bin" / "rope").is_file():
            return root

    raise RopePackageNotFoundError(
        f"no built rope-framework package found; set {PACKAGE_ROOT_ENV} or pass "
        f"package_root= explicitly (a rope-framework checkout with python/rope.py "
        f"and a built bin/rope+lib/librope.so or build/rope+build/librope.so)"
    )


def _env_build_dir() -> "Path | None":
    override = os.environ.get(BUILD_DIR_ENV)
    return Path(override) if override else None


def _load_rope_module(package_root: Path):
    """Imports package_root/python/rope.py directly from its file path."""
    rope_py = package_root / "python" / "rope.py"
    if not rope_py.is_file():
        raise RopePackageNotFoundError(f"{rope_py} not found")
    module_spec = importlib.util.spec_from_file_location("rope_framework_binding", rope_py)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _find_binary(lib_dir: Path, exe_dir: Path):
    """(exe, lib) if both exist under the given dirs, else None."""
    exe = exe_dir / "rope"
    lib = next(
        (c for c in (lib_dir / "librope.so", lib_dir / "librope.dylib", exe_dir / "librope.dll")
         if c.exists()),
        None,
    )
    return (exe, lib) if exe.is_file() and lib is not None else None


def _resolve_binary_paths(package_root: Path, *, build_dir: "Path | None" = None):
    """build_dir, if given, is searched directly (flat, then bin/+lib/) instead of package_root."""
    if build_dir is not None:
        build_dir = Path(build_dir)
        for lib_dir, exe_dir in ((build_dir, build_dir), (build_dir / "lib", build_dir / "bin")):
            found = _find_binary(lib_dir, exe_dir)
            if found is not None:
                return found
        raise RopePackageNotFoundError(f"no built rope binary/library found under {build_dir}")

    for lib_dir, exe_dir in ((package_root / "lib", package_root / "bin"),
                             (package_root / "build", package_root / "build")):
        found = _find_binary(lib_dir, exe_dir)
        if found is not None:
            return found
    raise RopePackageNotFoundError(f"no built rope binary/library found under {package_root}")


class ExportedDirModelInterface(ModelInterface):
    """Drives the real rope-framework binary/library against a candidate exported directory."""

    backend_name = "exported_dir"

    def __init__(self, exported_dir: Path, *, package_root: "Path | None" = None,
                 build_dir: "Path | None" = None, driver_path: "Path | None" = None):
        self.exported_dir = Path(exported_dir)
        self._grid = json.loads((self.exported_dir / "model_manifest.json").read_text())["grid"]

        build_dir = Path(build_dir) if build_dir else _env_build_dir()
        root = Path(package_root) if package_root else _discover_package_root(require_binary=build_dir is None)
        exe_path, lib_path = _resolve_binary_paths(root, build_dir=build_dir)
        rope_module = _load_rope_module(root)

        self._tmp_dir = tempfile.mkdtemp(prefix="rope_dev_tools_verify_")
        conf_path = Path(self._tmp_dir) / "rope.conf"
        # rope.conf paths resolve relative to its own dir, not cwd -- must write absolute paths.
        lines = [f"[paths]\nexported_dir = {self.exported_dir.resolve()}\n"]
        if driver_path:
            lines.append(f"driver_path = {Path(driver_path).resolve()}\n")
        # compute_uncertainty defaults true and costs ~10-15x a point forecast; unused, so disabled.
        lines.append("\n[forecast]\ncompute_uncertainty = false\n")
        conf_path.write_text("".join(lines))

        cache_path = str(Path(self._tmp_dir) / "forecast_grid.bin")
        self._rope = rope_module.Rope(
            lib_path=lib_path, exe_path=exe_path,
            cache_path=cache_path, config_path=conf_path,
        )

    @property
    def grid(self) -> dict:
        return self._grid

    def forecast(self, start: str, end: str) -> dict:
        horizon_hours = hours_between(start, end)
        result = self._rope.forecast(start, horizon_hours)
        self._rope.refresh()  # re-open the handle so it picks up this forecast's cache file
        return {"window_start": result["window_start"], "window_end": result["window_end"]}

    def query(self, time: str, lst: float, lat: float, alt_km: float) -> dict:
        return self._rope.get(time=time, lst=lst, lat=lat, alt_km=alt_km)

    def query_grid(self, time: str, alt_km: float) -> np.ndarray:
        n_lst, n_lat = self._grid["n_lst"], self._grid["n_lat"]
        lsts = np.linspace(0, 24, n_lst, endpoint=False)
        lats = np.linspace(self._grid["lat_min_deg"], self._grid["lat_max_deg"], n_lat)
        return self.query_grid_at(time, alt_km, lsts, lats)

    def query_grid_at(self, time: str, alt_km: float, lst_values: np.ndarray, lat_values: np.ndarray) -> np.ndarray:
        n_lst_q, n_lat_q = len(lst_values), len(lat_values)
        times_, lst_list, lat_list, alt_list = [], [], [], []
        for lst in lst_values:
            for lat in lat_values:
                times_.append(time)
                lst_list.append(float(lst))
                lat_list.append(float(lat))
                alt_list.append(alt_km)

        results = self._rope.get_batch(times_, lst_list, lat_list, alt_list)
        grid = np.zeros((n_lst_q, n_lat_q))
        idx = 0
        for i in range(n_lst_q):
            for j in range(n_lat_q):
                grid[i, j] = results[idx]["density"]
                idx += 1
        return grid

    def close(self) -> None:
        self._rope.shutdown()
