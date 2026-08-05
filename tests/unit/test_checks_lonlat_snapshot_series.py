"""lonlat_snapshot_series — per period, one forecast feeds static snapshots and a full-horizon animation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function
from rope_dev_tools.validation.checks.lonlat_snapshot_series import (
    _per_frame_statistics,
    replot_lonlat_snapshot_series,
)

_N_LON = 8
_LON_VALUES = np.linspace(0.0, 360.0, _N_LON, endpoint=False)
_GRID = {"n_lst": _N_LON, "n_lat": 6, "lat_min_deg": -80.0, "lat_max_deg": 80.0}


class _FakeModel:
    grid = _GRID

    def __init__(self):
        self.queried_lat_values = []
        self.queried_lst_values = []
        self.forecast_calls = []

    def forecast(self, start, end):
        self.forecast_calls.append((start, end))
        return {"window_start": start, "window_end": end}

    def query_grid_at(self, time, alt_km, lst_values, lat_values):
        self.queried_lat_values.append(list(lat_values))
        self.queried_lst_values.append(list(lst_values))
        return np.full((len(lst_values), len(lat_values)), 1.0e-12)


def _write_physics_npz(path):
    np.savez(
        path,
        times=np.array(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
        lon_values=_LON_VALUES, n_lat=6, lat_min_deg=-80.0, lat_max_deg=80.0,
        altitudes_km=np.array([400.0]),
        density=np.full((2, 1, _N_LON, 6), 1.0e-12),
    )


def _write_wide_physics_npz(path, *, lat_min, lat_max, n_lat):
    np.savez(
        path,
        times=np.array(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
        lon_values=_LON_VALUES, n_lat=n_lat, lat_min_deg=lat_min, lat_max_deg=lat_max,
        altitudes_km=np.array([400.0]),
        density=np.full((2, 1, _N_LON, n_lat), 1.0e-12),
    )


def _write_hourly_physics_npz(path, n_hours):
    base = datetime(2024, 1, 1)
    times = [(base + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") for h in range(n_hours)]
    np.savez(
        path,
        times=np.array(times),
        lon_values=_LON_VALUES, n_lat=6, lat_min_deg=-80.0, lat_max_deg=80.0,
        altitudes_km=np.array([400.0]),
        density=np.full((n_hours, 1, _N_LON, 6), 1.0e-12),
    )


def _write_two_day_physics_npz(path):
    # A single npz spanning 2 full days, as in the real suite where one hourly-density file is
    # shared across several 24h periods -- each period must only ever see its own [start, end)
    # slice, never the other day's hours.
    times = [f"2024-01-01 {h:02d}:00:00" for h in range(24)] + [f"2024-01-02 {h:02d}:00:00" for h in range(24)]
    np.savez(
        path,
        times=np.array(times),
        lon_values=_LON_VALUES, n_lat=6, lat_min_deg=-80.0, lat_max_deg=80.0,
        altitudes_km=np.array([400.0]),
        density=np.full((48, 1, _N_LON, 6), 1.0e-12),
    )


class _VaryingFakeModel:
    grid = _GRID

    def __init__(self, rope_value_by_alt):
        self.rope_value_by_alt = rope_value_by_alt

    def forecast(self, start, end):
        return {"window_start": start, "window_end": end}

    def query_grid_at(self, time, alt_km, lst_values, lat_values):
        return np.full((len(lst_values), len(lat_values)), self.rope_value_by_alt[alt_km])


def _write_varying_physics_npz(path):
    # altitude 400: phys density spans [1, 2]; altitude 500: phys density spans [100, 200].
    density = np.zeros((1, 2, _N_LON, 6))
    density[0, 0] = np.linspace(1.0, 2.0, _N_LON * 6).reshape(_N_LON, 6)
    density[0, 1] = np.linspace(100.0, 200.0, _N_LON * 6).reshape(_N_LON, 6)
    np.savez(
        path,
        times=np.array(["2024-01-01 00:00:00"]),
        lon_values=_LON_VALUES, n_lat=6, lat_min_deg=-80.0, lat_max_deg=80.0,
        altitudes_km=np.array([400.0, 500.0]),
        density=density,
    )


def _one_period(label="day1", start="2024-01-01 00:00:00", horizon_hours=1, utc_hours=(0,),
                npz="phys.npz", **extra):
    return {
        "label": label, "start": start, "horizon_hours": horizon_hours, "utc_hours": list(utc_hours),
        "physics_model_hourly_npz": npz, **extra,
    }


def test_lonlat_snapshot_series_standardizes_colorbar_per_altitude(tmp_path, monkeypatch):
    _write_varying_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")
    # rope's constant falls outside phys's own range at each altitude, so a correct shared
    # min/max must actually combine both sides, not just reflect physics data alone.
    model = _VaryingFakeModel(rope_value_by_alt={400.0: 3.0, 500.0: 50.0})

    captured = []
    import rope_dev_tools.validation.checks.lonlat_snapshot_series as mod

    def fake_lonlat_plot(panels, **kwargs):
        captured.append((kwargs.get("vmin"), kwargs.get("vmax")))
        return "plots/fake.png"

    monkeypatch.setattr(mod, "lonlat_plot", fake_lonlat_plot)

    fn(
        model, id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(include_animation=False)], altitudes_km=[400.0, 500.0],
    )

    # Two lonlat_plot calls per altitude (physics + rope), sharing one vmin/vmax each.
    alt400_calls, alt500_calls = captured[:2], captured[2:]
    assert alt400_calls[0] == alt400_calls[1] == pytest.approx((1.0, 3.0))
    assert alt500_calls[0] == alt500_calls[1] == pytest.approx((50.0, 200.0))
    assert alt400_calls[0] != alt500_calls[0]  # not constant across altitudes


def test_lonlat_snapshot_series_uses_suite_labels_in_titles(tmp_path, monkeypatch):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    captured = {"plot_suptitles": [], "anim_titles": []}
    import rope_dev_tools.validation.checks.lonlat_snapshot_series as mod

    def fake_lonlat_plot(panels, **kwargs):
        captured["plot_suptitles"].append(kwargs.get("suptitle"))
        return "plots/fake.png"

    def fake_lonlat_animation(panel_frames, **kwargs):
        captured["anim_titles"] = [p["title"] for p in panel_frames]
        return "plots/fake.gif"

    monkeypatch.setattr(mod, "lonlat_plot", fake_lonlat_plot)
    monkeypatch.setattr(mod, "lonlat_animation", fake_lonlat_animation)

    fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], altitudes_km=[400.0],
        physics_model_label="WAM", rope_model_label="ROPE-WAM-V1",
    )

    assert any("WAM" in s for s in captured["plot_suptitles"])
    assert any("ROPE-WAM-V1" in s for s in captured["plot_suptitles"])
    assert captured["anim_titles"] == ["WAM", "ROPE-WAM-V1"]


def test_lonlat_snapshot_series_writes_snapshots_and_animation(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(utc_hours=(0, 1))], altitudes_km=[400.0],
    )

    assert len(output["plots"]) == 3  # physics snapshot, rope snapshot, animation
    for plot in output["plots"]:
        assert (tmp_path / plot).is_file()
    for data_path in output["data"]:
        assert (tmp_path / data_path).is_file()


def test_lonlat_snapshot_series_queries_rope_lst_shifted_by_snapshot_time(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")
    model = _FakeModel()

    fn(
        model, id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(utc_hours=(0, 1))], altitudes_km=[400.0],
    )

    # utc_hours=[0, 1] on the same day -> the two snapshot queries' LST values for the same
    # longitude axis must differ by exactly 1 hour (mod 24), never converting WAM's data itself.
    lst_at_0, lst_at_1 = model.queried_lst_values[0], model.queried_lst_values[1]
    diffs = [(b - a) % 24.0 for a, b in zip(lst_at_0, lst_at_1)]
    np.testing.assert_allclose(diffs, 1.0, atol=1e-6)


def test_lonlat_snapshot_series_saved_npz_includes_lon_range(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(utc_hours=(0, 1))], altitudes_km=[400.0],
    )

    snap_npz = next(p for p in output["data"] if "_snapshots_" in p)
    with np.load(tmp_path / snap_npz) as npz:
        assert float(npz["lon_min_deg"]) == pytest.approx(_LON_VALUES.min())
        assert float(npz["lon_max_deg"]) == pytest.approx(_LON_VALUES.max())


def test_lonlat_snapshot_series_animation_over_72h_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period(horizon_hours=73, include_animation=True)], altitudes_km=[400.0],
        )


def test_lonlat_snapshot_series_missing_altitude_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period()], altitudes_km=[900.0],
        )


def test_lonlat_snapshot_series_restricts_to_rope_lat_bounds(tmp_path):
    # Physics data spans the full globe (-90..90); ROPE's own grid (_GRID) only covers -80..80.
    _write_wide_physics_npz(tmp_path / "phys.npz", lat_min=-90.0, lat_max=90.0, n_lat=7)
    fn = get_kind_function("lonlat_snapshot_series")
    model = _FakeModel()

    output = fn(
        model, id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(utc_hours=(0, 1))], altitudes_km=[400.0],
    )

    assert len(output["plots"]) == 3
    assert model.queried_lat_values  # actually got called
    for lat_values in model.queried_lat_values:
        assert all(_GRID["lat_min_deg"] <= v <= _GRID["lat_max_deg"] for v in lat_values)

    snap_npz = next(p for p in output["data"] if "_snapshots_" in p)
    with np.load(tmp_path / snap_npz) as npz:
        assert float(npz["lat_min_deg"]) >= _GRID["lat_min_deg"]
        assert float(npz["lat_max_deg"]) <= _GRID["lat_max_deg"]


def test_lonlat_snapshot_series_no_lat_overlap_raises(tmp_path):
    # Physics data only covers the polar cap; ROPE's grid (_GRID) stops at +/-80.
    _write_wide_physics_npz(tmp_path / "phys.npz", lat_min=85.0, lat_max=90.0, n_lat=3)
    fn = get_kind_function("lonlat_snapshot_series")

    with pytest.raises(ValueError, match="does not overlap"):
        fn(
            _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period()], altitudes_km=[400.0],
        )


def test_lonlat_snapshot_series_empty_periods_raises(tmp_path):
    fn = get_kind_function("lonlat_snapshot_series")
    with pytest.raises(ValueError):
        fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[], altitudes_km=[400.0])


def test_lonlat_snapshot_series_two_periods_do_not_collide(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(label="day1"), _one_period(label="day2")], altitudes_km=[400.0],
    )

    # 3 plots per period (physics snapshot, rope snapshot, animation), all distinct filenames.
    assert len(output["plots"]) == 6
    assert len(set(output["plots"])) == 6
    assert len(output["data"]) == 4  # 2 npz per period (snapshots + animation)
    assert len(set(output["data"])) == 4
    for path in output["plots"] + output["data"]:
        assert (tmp_path / path).is_file()


def test_lonlat_snapshot_series_restricts_to_period_time_window(tmp_path):
    # Regression: a shared multi-day npz must not leak day 2's *later* hours into a 24h period's
    # animation -- those fall outside that period's own model.forecast(start, end) window and a
    # real ROPE backend rejects querying them. The window itself is inclusive of its own end
    # boundary (ModelInterface.forecast docs: "[start, end]"), so the very first hour of day 2
    # (exactly 24h after start) is legitimately included; the second hour of day 2 is not.
    _write_two_day_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(horizon_hours=24, utc_hours=(0,))], altitudes_km=[400.0],
    )

    anim_npz = next(p for p in output["data"] if "_animation_" in p)
    with np.load(tmp_path / anim_npz) as npz:
        times = [str(t) for t in npz["times"]]
    assert len(times) == 25
    assert times[-1] == "2024-01-02 00:00:00"


def test_lonlat_snapshot_series_no_time_overlap_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")  # only has 2024-01-01 00:00 and 01:00
    fn = get_kind_function("lonlat_snapshot_series")

    with pytest.raises(ValueError, match="no timestamps"):
        fn(
            _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period(start="2025-06-01 00:00:00", horizon_hours=1, utc_hours=(0,))],
            altitudes_km=[400.0],
        )


def test_lonlat_snapshot_series_exactly_max_panels_stays_single_file(tmp_path):
    # At exactly the pagination limit (4), filenames stay unsuffixed -- unchanged from before
    # pagination existed.
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=4)
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(utc_hours=range(4), horizon_hours=4, include_animation=False)],
        altitudes_km=[400.0],
    )

    snapshot_plots = [p for p in output["plots"] if "_physics" in p or "_rope" in p]
    assert len(snapshot_plots) == 2
    assert all("_part" not in p for p in snapshot_plots)
    for p in snapshot_plots:
        assert (tmp_path / p).is_file()


def test_lonlat_snapshot_series_paginates_snapshots_over_max_panels(tmp_path, monkeypatch):
    # 6 hourly snapshots exceed the 4-per-plot cap -- must split into two pages (4 + 2) per
    # model (physics, rope), rather than cramming all 6 into one increasingly illegible row.
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=6)
    fn = get_kind_function("lonlat_snapshot_series")

    captured = []
    import rope_dev_tools.validation.checks.lonlat_snapshot_series as mod

    def fake_lonlat_plot(panels, **kwargs):
        captured.append(len(panels))
        return kwargs.get("out_path")

    monkeypatch.setattr(mod, "lonlat_plot", fake_lonlat_plot)

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(utc_hours=range(6), horizon_hours=6, include_animation=False)],
        altitudes_km=[400.0],
    )

    assert sorted(captured) == [2, 2, 4, 4]  # 2 pages x (physics, rope)
    assert sum(1 for p in output["plots"] if p.endswith("_part1.png")) == 2
    assert sum(1 for p in output["plots"] if p.endswith("_part2.png")) == 2


def test_replot_lonlat_snapshot_series_paginates_snapshots(tmp_path):
    n = 6
    times = np.array([f"2024-01-01 {h:02d}:00:00" for h in range(n)])
    grids = np.full((n, _N_LON, 6), 1.0e-12)
    npz_path = "validation_data/snap_test_snapshots_day1_400.0km.npz"
    loaded = {
        npz_path: {
            "times": times, "physics_density": grids, "rope_density": grids,
            "lat_min_deg": np.array(-80.0), "lat_max_deg": np.array(80.0),
            "lon_min_deg": np.array(0.0), "lon_max_deg": np.array(350.0),
        }
    }

    plots = replot_lonlat_snapshot_series(loaded, id="snap_test", out_dir=tmp_path)

    assert any(p.endswith("_part1.png") for p in plots)
    assert any(p.endswith("_part2.png") for p in plots)
    for p in plots:
        assert (tmp_path / p).is_file()


def test_lonlat_snapshot_series_spans_multiple_days_for_long_horizon(tmp_path):
    # A ~2-day period auto-covers both calendar days -- utc_hours=[0, 12] x 2 days = 4 snapshots,
    # matching the "hourly snapshots over a 48h storm" use case without a separate "days" list.
    # horizon_hours=47 (not 48) keeps the window's end inside day 2 (2024-01-02 23:00), so this
    # test isolates the "N full days" case cleanly -- see the trailing-partial-day test below for
    # the exact-48h edge where the window's end touches a 3rd calendar date.
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=48)
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(horizon_hours=47, utc_hours=(0, 12), include_animation=False)],
        altitudes_km=[400.0],
    )

    snap_npz = next(p for p in output["data"] if "_snapshots_" in p)
    with np.load(tmp_path / snap_npz) as npz:
        times = [str(t) for t in npz["times"]]
    assert times == [
        "2024-01-01 00:00:00", "2024-01-01 12:00:00",
        "2024-01-02 00:00:00", "2024-01-02 12:00:00",
    ]


def test_lonlat_snapshot_series_trailing_partial_day_skips_out_of_window_hours(tmp_path):
    # horizon_hours=24 starting at midnight -> end is exactly the start of day 2 (hour 0 only).
    # utc_hours=[0..23] must not error on day 2's hours 1-23 (outside the period's own window) --
    # those are silently not applicable, not a data gap.
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=48)
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(horizon_hours=24, utc_hours=range(24), include_animation=False)],
        altitudes_km=[400.0],
    )

    snap_npz = next(p for p in output["data"] if "_snapshots_" in p)
    with np.load(tmp_path / snap_npz) as npz:
        times = [str(t) for t in npz["times"]]
    assert len(times) == 25  # day 1's 24 hours + day 2's hour 0 (the inclusive window boundary)
    assert times[-1] == "2024-01-02 00:00:00"


def test_lonlat_snapshot_series_no_utc_hours_in_window_raises(tmp_path):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=48)
    fn = get_kind_function("lonlat_snapshot_series")

    with pytest.raises(ValueError, match="utc_hours"):
        fn(
            _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period(horizon_hours=1, utc_hours=(12,), include_animation=False)],
            altitudes_km=[400.0],
        )


def test_lonlat_snapshot_series_statistics_nested_by_period(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(label="day1"), _one_period(label="day2")], altitudes_km=[400.0],
        statistics=["bias"],
    )

    assert set(output["statistics"]) == {"day1", "day2"}
    assert "400.0km" in output["statistics"]["day1"]


def test_lonlat_snapshot_series_default_start_delta_produces_unsuffixed_names(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("lonlat_snapshot_series")

    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period(utc_hours=(0, 1))], altitudes_km=[400.0],
    )

    assert all("_delta" not in p for p in output["plots"])
    assert all("_delta" not in p for p in output["data"])


def test_lonlat_snapshot_series_start_delta_shifts_forecast_start(tmp_path):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=6)
    fn = get_kind_function("lonlat_snapshot_series")
    model = _FakeModel()

    period = _one_period(horizon_hours=5, utc_hours=(0,), start_deltas=[-2, 0])
    fn(model, id="snap_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period], altitudes_km=[400.0])

    assert ("2023-12-31 22:00:00", "2024-01-01 05:00:00") in model.forecast_calls
    assert ("2024-01-01 00:00:00", "2024-01-01 05:00:00") in model.forecast_calls


def test_lonlat_snapshot_series_multiple_deltas_produce_suffixed_plots(tmp_path):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=4)
    fn = get_kind_function("lonlat_snapshot_series")

    period = _one_period(horizon_hours=3, utc_hours=(0, 2), start_deltas=[0, 2], include_animation=False)
    output = fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
                periods=[period], altitudes_km=[400.0])

    plots = output["plots"]
    assert any(p.endswith("_physics.png") for p in plots)  # physics stays unsuffixed
    assert any("_rope_delta+0.png" in p for p in plots)
    assert any("_rope_delta+2.png" in p for p in plots)
    for p in plots:
        assert (tmp_path / p).is_file()


def test_lonlat_snapshot_series_widest_delta_gives_physics_full_window(tmp_path, monkeypatch):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=4)
    fn = get_kind_function("lonlat_snapshot_series")

    captured = []
    import rope_dev_tools.validation.checks.lonlat_snapshot_series as mod

    def fake_lonlat_plot(panels, **kwargs):
        captured.append((kwargs.get("out_path"), len(panels)))
        return kwargs.get("out_path")

    monkeypatch.setattr(mod, "lonlat_plot", fake_lonlat_plot)

    period = _one_period(horizon_hours=3, utc_hours=(0, 2), start_deltas=[0, 2], include_animation=False)
    fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period], altitudes_km=[400.0])

    by_path = {path: n for path, n in captured}
    physics_path = next(p for p in by_path if "_physics" in p)
    rope_delta0_path = next(p for p in by_path if "_rope_delta+0" in p)
    rope_delta2_path = next(p for p in by_path if "_rope_delta+2" in p)
    # widest_delta (min=0) gives physics the full 2-panel window; delta=+2 narrows to 1 panel
    # (hour 0 falls before its query_start_dt of 02:00), while delta=+0's own rope stays at 2.
    assert by_path[physics_path] == 2
    assert by_path[rope_delta0_path] == 2
    assert by_path[rope_delta2_path] == 1


def test_lonlat_snapshot_series_multiple_deltas_nested_statistics(tmp_path):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=4)
    fn = get_kind_function("lonlat_snapshot_series")

    period = _one_period(horizon_hours=3, utc_hours=(0, 2), start_deltas=[0, 2], include_animation=False)
    output = fn(
        _FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[period], altitudes_km=[400.0], statistics=["bias"],
    )

    snapshot_stats = output["statistics"]["day1"]["400.0km"]["snapshot"]
    assert set(snapshot_stats) == {"delta_+0h", "delta_+2h"}


def test_lonlat_snapshot_series_negative_delta_does_not_count_toward_max_animation_hours(tmp_path):
    # A negative delta only extends the warm-up *before* the animated window -- frames are always
    # clamped to query_start_dt (== the period's own start, for delta <= 0), so however far back
    # the warm-up reaches, it must never inflate the animated/rendered duration. horizon_hours=48
    # with delta=-48 would exceed the 72h cap under the old (buggy) "full simulation duration"
    # accounting (48 - (-48) = 96h) -- it must not raise now.
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=48)
    fn = get_kind_function("lonlat_snapshot_series")

    period = _one_period(horizon_hours=48, utc_hours=(0,), start_deltas=[-48],
                          include_animation=True, include_snapshots=False)
    output = fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
                periods=[period], altitudes_km=[400.0])
    assert any(p.endswith(".gif") for p in output["plots"])


def test_lonlat_snapshot_series_large_horizon_hours_over_max_animation_raises(tmp_path):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=2)
    fn = get_kind_function("lonlat_snapshot_series")

    # horizon_hours alone (independent of delta) still bounds the animated window.
    period = _one_period(horizon_hours=73, utc_hours=(0,), start_deltas=[0], include_animation=True)
    with pytest.raises(ValueError, match="animated window"):
        fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
           periods=[period], altitudes_km=[400.0])


def test_lonlat_snapshot_series_multiple_deltas_animation_filenames(tmp_path):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=4)
    fn = get_kind_function("lonlat_snapshot_series")

    period = _one_period(horizon_hours=3, utc_hours=(0,), start_deltas=[0, 2], include_snapshots=False)
    output = fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
                periods=[period], altitudes_km=[400.0])

    assert any(p.endswith("_animation_delta+0.gif") for p in output["plots"])
    assert any(p.endswith("_animation_delta+2.gif") for p in output["plots"])
    for p in output["plots"]:
        assert (tmp_path / p).is_file()


def test_lonlat_snapshot_series_plot_stats_requires_statistics(tmp_path):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=2)
    fn = get_kind_function("lonlat_snapshot_series")

    period = _one_period(horizon_hours=1, utc_hours=(0,), plot_stats=True)
    with pytest.raises(ValueError, match="plot_stats"):
        fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
           periods=[period], altitudes_km=[400.0])


def test_lonlat_snapshot_series_plot_stats_passes_per_frame_series_to_animation(tmp_path, monkeypatch):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=4)
    fn = get_kind_function("lonlat_snapshot_series")

    captured = []
    import rope_dev_tools.validation.checks.lonlat_snapshot_series as mod

    def fake_lonlat_animation(panel_frames, **kwargs):
        captured.append(kwargs.get("stats_series"))
        return kwargs.get("out_path")

    monkeypatch.setattr(mod, "lonlat_animation", fake_lonlat_animation)

    period = _one_period(horizon_hours=3, utc_hours=(0,), plot_stats=True, include_snapshots=False)
    fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
       periods=[period], altitudes_km=[400.0], statistics=["bias", "rmse"])

    assert len(captured) == 1
    stats_series = captured[0]
    assert set(stats_series) == {"bias", "rmse"}
    assert len(stats_series["bias"]) == 4  # one value per animation frame (n_hours=4, step=1)


def test_lonlat_snapshot_series_plot_stats_false_omits_stats_series(tmp_path, monkeypatch):
    _write_hourly_physics_npz(tmp_path / "phys.npz", n_hours=2)
    fn = get_kind_function("lonlat_snapshot_series")

    captured = []
    import rope_dev_tools.validation.checks.lonlat_snapshot_series as mod

    def fake_lonlat_animation(panel_frames, **kwargs):
        captured.append(kwargs.get("stats_series"))
        return kwargs.get("out_path")

    monkeypatch.setattr(mod, "lonlat_animation", fake_lonlat_animation)

    period = _one_period(horizon_hours=1, utc_hours=(0,), include_snapshots=False)
    fn(_FakeModel(), id="snap_test", out_dir=tmp_path, suite_dir=tmp_path,
       periods=[period], altitudes_km=[400.0], statistics=["bias"])

    assert captured == [None]


def test_per_frame_statistics_computes_one_value_per_frame():
    rope_frames = [np.full((2, 2), 1.0), np.full((2, 2), 2.0)]
    phys_frames = [np.full((2, 2), 1.0), np.full((2, 2), 1.0)]
    result = _per_frame_statistics(rope_frames, phys_frames, ["bias"])
    # bias is a percent (0 = unbiased): frame 0 matches exactly (0%), frame 1 is double (+100%).
    np.testing.assert_allclose(result["bias"], [0.0, 100.0])
