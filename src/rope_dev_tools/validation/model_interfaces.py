"""Two interchangeable ModelInterface backends behind one small interface.

Wrapper mode drives a dev-supplied callable directly (fast, pre-export,
in-memory). Exported-dir mode drives the real rope-framework binary/library
against a candidate exported directory (full integration-level check,
including the actual C++ runtime-compat and pipeline-load path).

Each check-kind function calls forecast()/query()/query_grid() itself,
whenever and however often it needs to — there's no suite-level forecast
orchestration here; that decision belongs entirely to each kind.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rope_dev_tools.grid import ALT_MAX_KM, ALT_MIN_KM, GRID_ALT, GRID_LAT, GRID_LST, LAT_MAX, LAT_MIN
from rope_dev_tools.validation.time_utils import hours_between, parse_time

PACKAGE_ROOT_ENV = "ROPE_PACKAGE_ROOT"


class ModelInterface(ABC):
    @abstractmethod
    def forecast(self, start: str, end: str) -> dict:
        """Forecasts [start, end]. Returns {"window_start": <str>, "window_end": <str>}
        — the actual usable query window, which does NOT start at `start`
        itself: some history warm-up is consumed first, so the earliest
        valid query time is later than `start`. Callers must query within
        the returned window, not assume `start`/`end` are themselves queryable.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, time: str, lst: float, lat: float, alt_km: float) -> dict:
        raise NotImplementedError

    @abstractmethod
    def query_grid(self, time: str, alt_km: float) -> np.ndarray:
        """Returns a (GRID_LST, GRID_LAT) density array at the given time/altitude."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook; a no-op unless a backend owns a process/handle."""
        return None


# ---------------------------------------------------------------------------
# Wrapper mode
# ---------------------------------------------------------------------------

@dataclass
class WrapperRequest:
    start: str
    end: str


@dataclass
class WrapperResponse:
    times: list             # ISO timestamps, length T
    density: np.ndarray      # (T, GRID_LST, GRID_LAT, GRID_ALT)
    uncertainty: np.ndarray  # same shape


WrapperFn = Callable[[WrapperRequest], WrapperResponse]


def _lst_index(lst: float) -> int:
    return int(round((lst % 24.0) / 24.0 * GRID_LST)) % GRID_LST


def _lat_index(lat: float) -> int:
    lat = min(max(lat, LAT_MIN), LAT_MAX)
    frac = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    return int(round(frac * (GRID_LAT - 1)))


def _alt_index(alt_km: float) -> int:
    alt_km = min(max(alt_km, ALT_MIN_KM), ALT_MAX_KM)
    frac = (alt_km - ALT_MIN_KM) / (ALT_MAX_KM - ALT_MIN_KM)
    return int(round(frac * (GRID_ALT - 1)))


def _nearest_time_index(times: list, time: str) -> int:
    target = parse_time(time)
    diffs = [abs((parse_time(t) - target).total_seconds()) for t in times]
    return diffs.index(min(diffs))


class WrapperModelInterface(ModelInterface):
    """Drives a dev-supplied callable directly — one call per forecast(),
    matching the real system's forecast-once/query-many split. Uses
    nearest-grid-cell lookup (not the production trilinear/log-space
    interpolation rope-framework implements) since this is a fast pre-export
    sanity check against the dev's own in-memory model, not a
    re-implementation of the C++ interpolator.
    """

    def __init__(self, wrapper_fn: WrapperFn):
        self._wrapper_fn = wrapper_fn
        self._response: "WrapperResponse | None" = None

    def forecast(self, start: str, end: str) -> dict:
        self._response = self._wrapper_fn(WrapperRequest(start=start, end=end))
        return {"window_start": self._response.times[0], "window_end": self._response.times[-1]}

    def query(self, time: str, lst: float, lat: float, alt_km: float) -> dict:
        if self._response is None:
            raise RuntimeError("forecast() must be called before query()")
        ti = _nearest_time_index(self._response.times, time)
        li, ai, alti = _lst_index(lst), _lat_index(lat), _alt_index(alt_km)
        density = float(self._response.density[ti, li, ai, alti])
        uncertainty = float(self._response.uncertainty[ti, li, ai, alti])
        return {"density": density, "uncertainty": uncertainty}

    def query_grid(self, time: str, alt_km: float) -> np.ndarray:
        if self._response is None:
            raise RuntimeError("forecast() must be called before query_grid()")
        ti = _nearest_time_index(self._response.times, time)
        alti = _alt_index(alt_km)
        return np.asarray(self._response.density[ti, :, :, alti])


# ---------------------------------------------------------------------------
# Exported-directory mode
# ---------------------------------------------------------------------------

class RopePackageNotFoundError(RuntimeError):
    pass


def _discover_package_root() -> Path:
    override = os.environ.get(PACKAGE_ROOT_ENV)
    if override:
        return Path(override)

    candidates = [Path.cwd() / "rope-framework", Path.cwd().parent / "rope-framework"]
    for root in candidates:
        if (root / "python" / "rope.py").is_file() and (
            (root / "build" / "rope").is_file() or (root / "bin" / "rope").is_file()
        ):
            return root

    raise RopePackageNotFoundError(
        f"no built rope-framework package found; set {PACKAGE_ROOT_ENV} or pass "
        f"package_root= explicitly (a rope-framework checkout with python/rope.py "
        f"and a built bin/rope+lib/librope.so or build/rope+build/librope.so)"
    )


def _load_rope_module(package_root: Path):
    rope_py = package_root / "python" / "rope.py"
    if not rope_py.is_file():
        raise RopePackageNotFoundError(f"{rope_py} not found")
    module_spec = importlib.util.spec_from_file_location("rope_framework_binding", rope_py)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _resolve_binary_paths(package_root: Path):
    for lib_dir, exe_dir in ((package_root / "lib", package_root / "bin"),
                             (package_root / "build", package_root / "build")):
        exe = exe_dir / "rope"
        lib = next(
            (c for c in (lib_dir / "librope.so", lib_dir / "librope.dylib", exe_dir / "librope.dll")
             if c.exists()),
            None,
        )
        if exe.is_file() and lib is not None:
            return exe, lib
    raise RopePackageNotFoundError(f"no built rope binary/library found under {package_root}")


class ExportedDirModelInterface(ModelInterface):
    """Drives the real rope-framework binary/library against a candidate
    exported directory, via rope-framework/python/rope.py's Rope class —
    full integration-level check including the actual C++ runtime-compat
    and pipeline-load path.
    """

    def __init__(self, exported_dir: Path, *, package_root: "Path | None" = None,
                 driver_path: "Path | None" = None):
        self.exported_dir = Path(exported_dir)
        root = Path(package_root) if package_root else _discover_package_root()
        exe_path, lib_path = _resolve_binary_paths(root)
        rope_module = _load_rope_module(root)

        self._tmp_dir = tempfile.mkdtemp(prefix="rope_dev_tools_verify_")
        conf_path = Path(self._tmp_dir) / "rope.conf"
        lines = [f"[paths]\nexported_dir = {self.exported_dir}\n"]
        if driver_path:
            lines.append(f"driver_path = {driver_path}\n")
        conf_path.write_text("".join(lines))

        sock_path = str(Path(self._tmp_dir) / "rope.sock")
        self._rope = rope_module.Rope(
            lib_path=lib_path, exe_path=exe_path,
            socket_path=sock_path, config_path=conf_path,
        )

    def forecast(self, start: str, end: str) -> dict:
        horizon_hours = hours_between(start, end)
        result = self._rope.forecast(start, horizon_hours)
        return {"window_start": result["window_start"], "window_end": result["window_end"]}

    def query(self, time: str, lst: float, lat: float, alt_km: float) -> dict:
        return self._rope.get(time=time, lst=lst, lat=lat, alt_km=alt_km)

    def query_grid(self, time: str, alt_km: float) -> np.ndarray:
        lsts = np.linspace(0, 24, GRID_LST, endpoint=False)
        lats = np.linspace(LAT_MIN, LAT_MAX, GRID_LAT)
        times_, lst_list, lat_list, alt_list = [], [], [], []
        for lst in lsts:
            for lat in lats:
                times_.append(time)
                lst_list.append(float(lst))
                lat_list.append(float(lat))
                alt_list.append(alt_km)

        results = self._rope.get_batch(times_, lst_list, lat_list, alt_list)
        grid = np.zeros((GRID_LST, GRID_LAT))
        idx = 0
        for i in range(GRID_LST):
            for j in range(GRID_LAT):
                grid[i, j] = results[idx]["density"]
                idx += 1
        return grid

    def close(self) -> None:
        self._rope.shutdown()
