"""altitude_profile — global-average density vs altitude at selected time slices."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function
from rope_dev_tools.validation.checks.altitude_profile import replot_altitude_profile

_N_LON = 8
_LON_VALUES = np.linspace(0.0, 360.0, _N_LON, endpoint=False)
_N_LAT = 6
_ALTITUDES = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 450.0, 600.0])
_GRID = {"n_lst": _N_LON, "n_lat": _N_LAT, "lat_min_deg": -80.0, "lat_max_deg": 80.0,
         "n_alt": len(_ALTITUDES), "alt_min_km": 100.0, "alt_max_km": 1000.0}


class _FakeModel:
    """Returns a constant density at every grid point."""
    grid = _GRID

    def __init__(self, density_value=1.0e-12):
        self.forecast_calls = []
        self._density_value = density_value

    def forecast(self, start, end, *, compute_uncertainty=False):
        self.forecast_calls.append((start, end))

    def query_grid_at(self, time, alt_km, lst_values, lat_values, *, include_uncertainty=False):
        return np.full((len(lst_values), len(lat_values)), self._density_value)


class _VaryingFakeModel:
    """Returns altitude-dependent density."""
    grid = _GRID

    def __init__(self, density_by_alt):
        self.forecast_calls = []
        self._density_by_alt = density_by_alt

    def forecast(self, start, end, *, compute_uncertainty=False):
        self.forecast_calls.append((start, end))

    def query_grid_at(self, time, alt_km, lst_values, lat_values, *, include_uncertainty=False):
        return np.full((len(lst_values), len(lat_values)), self._density_by_alt[alt_km])


def _write_physics_npz(path, *, n_hours=2, density_value=1.0e-12, altitudes=_ALTITUDES):
    """Writes a physics NPZ with n_hours hourly timestamps starting 2024-01-01 00:00."""
    base = datetime(2024, 1, 1)
    times = [(base + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") for h in range(n_hours)]
    np.savez(
        path,
        times=np.array(times),
        lon_values=_LON_VALUES, n_lat=_N_LAT, lat_min_deg=-80.0, lat_max_deg=80.0,
        altitudes_km=altitudes,
        density=np.full((n_hours, len(altitudes), _N_LON, _N_LAT), density_value),
    )


def _one_period(label="day1", start="2024-01-01 00:00:00", horizon_hours=1, utc_hours=(0,),
                npz="phys.npz", **extra):
    """Builds a single period dict for the altitude_profile check."""
    return {
        "label": label, "start": start, "horizon_hours": horizon_hours, "utc_hours": list(utc_hours),
        "physics_model_hourly_npz": npz, **extra,
    }


# --- basic functionality ---

def test_altitude_profile_produces_plots_and_data(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")
    output = fn(
        _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()],
    )
    assert len(output["plots"]) == 1
    for plot in output["plots"]:
        assert (tmp_path / plot).is_file()
    assert len(output["data"]) == 1
    for d in output["data"]:
        assert (tmp_path / d).is_file()


def test_altitude_profile_queries_each_physics_altitude(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")

    class _TrackerModel(_FakeModel):
        def __init__(self):
            super().__init__()
            self.queried_alts = []

        def query_grid_at(self, time, alt_km, lst_values, lat_values, **_):
            self.queried_alts.append(alt_km)
            return super().query_grid_at(time, alt_km, lst_values, lat_values)

    model = _TrackerModel()
    fn(model, id="alt_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()])
    assert sorted(model.queried_alts) == sorted(_ALTITUDES.tolist())


def test_altitude_profile_multiple_utc_hours_produce_multiple_panels(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz", n_hours=25)
    fn = get_kind_function("altitude_profile")

    captured = []
    import rope_dev_tools.validation.checks.altitude_profile as mod
    orig = mod._altitude_profile_plot

    def fake_plot(panels, **kwargs):
        captured.append(len(panels))
        return orig(panels, **kwargs)

    mod._altitude_profile_plot = fake_plot
    try:
        fn(
            _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period(horizon_hours=24, utc_hours=(0, 6, 12, 18))],
        )
    finally:
        mod._altitude_profile_plot = orig
    assert captured == [4]


def test_altitude_profile_empty_periods_raises(tmp_path):
    fn = get_kind_function("altitude_profile")
    with pytest.raises(ValueError, match="periods is empty"):
        fn(_FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[])


def test_altitude_profile_missing_time_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz", n_hours=25)
    fn = get_kind_function("altitude_profile")
    # utc_hour 3 exists in the day range but is not in the NPZ (which has hourly 0-24).
    # Actually all 25 hours are in the npz, so test a different scenario:
    # npz only has hours 0 and 1, horizon covers hours 0-23, request hour 6 which is in
    # the day range but missing from the npz.
    _write_physics_npz(tmp_path / "sparse.npz", n_hours=2)
    with pytest.raises(ValueError, match="missing from"):
        fn(
            _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period(npz="sparse.npz", horizon_hours=24, utc_hours=(0, 6))],
        )


# --- statistics ---

def test_altitude_profile_computes_statistics_when_requested(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz", density_value=2.0e-12)
    fn = get_kind_function("altitude_profile")
    model = _FakeModel(density_value=1.8e-12)
    output = fn(
        model, id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], statistics=["bias", "rmse", "std"],
    )
    assert "statistics" in output
    stats = output["statistics"]
    assert "day1" in stats


def test_altitude_profile_no_statistics_key_when_not_requested(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")
    output = fn(
        _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()],
    )
    assert "statistics" not in output


# --- start_deltas ---

def test_altitude_profile_start_deltas_produce_separate_plots(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz", n_hours=25)
    fn = get_kind_function("altitude_profile")
    output = fn(
        _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(horizon_hours=24, utc_hours=(0,), start_deltas=[0, -12])],
    )
    assert len(output["plots"]) == 2


def test_altitude_profile_start_delta_shifts_forecast_window(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz", n_hours=25)
    fn = get_kind_function("altitude_profile")
    model = _FakeModel()
    fn(
        model, id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(horizon_hours=1, utc_hours=(0,), start_deltas=[-12])],
    )
    assert len(model.forecast_calls) == 1
    start, end = model.forecast_calls[0]
    assert start == "2023-12-31 12:00:00"


# --- lat clipping ---

def test_altitude_profile_clips_physics_lats_to_rope_grid(tmp_path):
    np.savez(
        tmp_path / "phys.npz",
        times=np.array(["2024-01-01 00:00:00"]),
        lon_values=_LON_VALUES, n_lat=18, lat_min_deg=-90.0, lat_max_deg=90.0,
        altitudes_km=np.array([300.0]),
        density=np.full((1, 1, _N_LON, 18), 1.0e-12),
    )
    fn = get_kind_function("altitude_profile")

    class _LatTracker(_FakeModel):
        def __init__(self):
            super().__init__()
            self.queried_lats = []

        def query_grid_at(self, time, alt_km, lst_values, lat_values, **_):
            self.queried_lats.append(list(lat_values))
            return super().query_grid_at(time, alt_km, lst_values, lat_values)

    model = _LatTracker()
    fn(model, id="alt_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()])
    for lats in model.queried_lats:
        assert min(lats) >= -80.0
        assert max(lats) <= 80.0


# --- suite labels ---

def test_altitude_profile_uses_suite_labels(tmp_path, monkeypatch):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")

    captured = []
    import rope_dev_tools.validation.checks.altitude_profile as mod

    def fake_plot(panels, **kwargs):
        for p in panels:
            captured.extend(p["series"].keys())
        return "plots/fake.png"

    monkeypatch.setattr(mod, "_altitude_profile_plot", fake_plot)
    fn(
        _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()],
        physics_model_label="WAM", rope_model_label="ROPE-WAM-V1",
    )
    assert "WAM" in captured
    assert "ROPE-WAM-V1" in captured


# --- replot ---

def test_replot_altitude_profile_regenerates_from_saved_npz(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")
    output = fn(
        _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], statistics=["bias", "rmse", "std"],
    )

    for p in (tmp_path / "plots").iterdir():
        p.unlink()

    loaded = {}
    for d in output["data"]:
        with np.load(tmp_path / d) as npz:
            loaded[d] = dict(npz)
    plots = replot_altitude_profile(loaded, id="alt_test", out_dir=tmp_path)
    assert len(plots) == 1
    for p in plots:
        assert (tmp_path / p).is_file()


# --- saved data roundtrip ---

def test_altitude_profile_saved_npz_contains_profiles(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz", density_value=5.0e-12)
    fn = get_kind_function("altitude_profile")
    model = _VaryingFakeModel({alt: 4.0e-12 for alt in _ALTITUDES})
    output = fn(
        model, id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()],
    )
    with np.load(tmp_path / output["data"][0]) as npz:
        assert "physics_profiles" in npz
        assert "rope_profiles" in npz
        assert "altitudes_km" in npz
        np.testing.assert_allclose(npz["physics_profiles"][0], 5.0e-12)
        np.testing.assert_allclose(npz["rope_profiles"][0], 4.0e-12)


# --- altitude cutouts ---

def test_altitude_cutouts_produce_extra_plots(tmp_path):
    """Each cutout generates one additional plot per period/delta."""
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")
    cutouts = [{"label": "low_alt", "min_km": 100, "max_km": 300}]
    output = fn(
        _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], altitude_cutouts=cutouts,
    )
    assert len(output["plots"]) == 2
    assert any("low_alt" in p for p in output["plots"])
    for p in output["plots"]:
        assert (tmp_path / p).is_file()


def test_altitude_cutouts_filter_altitude_range(tmp_path):
    """Cutout panels only contain altitudes within [min_km, max_km]."""
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")

    captured_panels = []
    import rope_dev_tools.validation.checks.altitude_profile as mod
    orig = mod._altitude_profile_plot

    def spy_plot(panels, **kwargs):
        captured_panels.append(panels)
        return orig(panels, **kwargs)

    mod._altitude_profile_plot = spy_plot
    try:
        fn(
            _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period()],
            altitude_cutouts=[{"label": "low", "min_km": 100, "max_km": 300}],
        )
    finally:
        mod._altitude_profile_plot = orig

    # First call is the full-range plot, second is the cutout.
    assert len(captured_panels) == 2
    cutout_panel = captured_panels[1][0]
    _, alts = list(cutout_panel["series"].values())[0]
    assert alts.min() >= 100.0
    assert alts.max() <= 300.0
    assert len(alts) == 5  # 100, 150, 200, 250, 300


def test_replot_altitude_cutouts(tmp_path):
    """replot_altitude_profile also produces cutout plots when altitude_cutouts is passed."""
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("altitude_profile")
    output = fn(
        _FakeModel(), id="alt_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], statistics=["bias", "rmse", "std"],
    )

    for p in (tmp_path / "plots").iterdir():
        p.unlink()

    loaded = {}
    for d in output["data"]:
        with np.load(tmp_path / d) as npz:
            loaded[d] = dict(npz)

    cutouts = [{"label": "low_alt", "min_km": 100, "max_km": 300}]
    plots = replot_altitude_profile(loaded, id="alt_test", out_dir=tmp_path, altitude_cutouts=cutouts)
    assert len(plots) == 2
    assert any("low_alt" in p for p in plots)
    for p in plots:
        assert (tmp_path / p).is_file()
