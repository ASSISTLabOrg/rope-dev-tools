"""satellite_orbit_density — satellite/physics/rope density along a track, any number of periods."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function
from rope_dev_tools.validation.checks.satellite_orbit_density import _orbit_average, replot_satellite_orbit_density


class _FakeModel:
    def __init__(self):
        self.forecast_calls = []

    def forecast(self, start, end):
        self.forecast_calls.append((start, end))
        return {"window_start": start, "window_end": end}

    def query(self, time, lst, lat, alt_km):
        return {"density": 1.0e-12, "uncertainty": 1.0e-13}


def _write_track_csv(path, density, *, rows=None):
    rows = rows or [
        ("2024-01-01 00:00:00", 12.0, 0.0, 400.0, density),
        ("2024-01-01 01:00:00", 13.0, 5.0, 400.0, density),
    ]
    lines = ["datetime,lst,lat,alt_km,density"]
    for dt, lst, lat, alt_km, d in rows:
        lines.append(f"{dt},{lst},{lat},{alt_km},{d}")
    path.write_text("\n".join(lines) + "\n")


def _one_period(label="p1", start="2024-01-01 00:00:00", end="2024-01-01 02:00:00",
                sat="sat.csv", phys="phys.csv"):
    return {"label": label, "start": start, "end": end,
            "satellite_track_csv": sat, "physics_model_track_csv": phys}


def test_satellite_orbit_density_writes_plot_data_and_value(tmp_path):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()],
    )

    assert (tmp_path / output["plots"][0]).is_file()
    assert (tmp_path / output["data"][0]).is_file()
    # statistics/value are always nested one level deeper by start_delta, even for the default
    # [0]-only case -- a uniform output shape regardless of whether start_deltas is actually used.
    assert output["per_period"]["p1"]["delta_+0h"]["value"] == pytest.approx(0.0, abs=1e-20)
    assert output["per_period"]["p1"]["delta_+0h"]["passed"] is None
    assert output["passed"] is None
    assert "statistics" not in output


def test_satellite_orbit_density_row_mismatch_raises(tmp_path):
    (tmp_path / "sat.csv").write_text(
        "datetime,lst,lat,alt_km,density\n2024-01-01 00:00:00,12.0,0.0,400.0,1.0e-12\n"
    )
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[_one_period()],
        )


def test_satellite_orbit_density_computes_requested_statistics(tmp_path):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], statistics=["bias"],
    )

    assert set(output["statistics"]["p1"]["delta_+0h"]) == {
        "rope_vs_satellite", "physics_vs_satellite", "rope_vs_physics_model",
    }


def test_satellite_orbit_density_empty_periods_raises(tmp_path):
    fn = get_kind_function("satellite_orbit_density")
    with pytest.raises(ValueError):
        fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[])


def test_satellite_orbit_density_filters_rows_outside_period(tmp_path):
    rows = [
        ("2023-12-31 23:00:00", 12.0, 0.0, 400.0, 9.0e-12),  # outside [start, end) -- excluded
        ("2024-01-01 00:00:00", 12.0, 0.0, 400.0, 1.0e-12),
        ("2024-01-01 01:00:00", 13.0, 5.0, 400.0, 1.0e-12),
        ("2024-01-02 00:00:00", 12.0, 0.0, 400.0, 9.0e-12),  # outside [start, end) -- excluded
    ]
    _write_track_csv(tmp_path / "sat.csv", None, rows=rows)
    _write_track_csv(tmp_path / "phys.csv", None, rows=rows)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()],
    )

    import pandas as pd
    comparison = pd.read_csv(tmp_path / output["data"][0])
    assert len(comparison) == 2
    assert set(comparison["satellite_density"]) == {1.0e-12}


def test_satellite_orbit_density_two_periods_threshold_split(tmp_path):
    # p1: rope (1.0e-12 via _FakeModel) matches satellite exactly -> passes a tight threshold
    _write_track_csv(tmp_path / "sat_pass.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys_pass.csv", 0.9e-12)
    # p2: satellite density far from rope's fixed 1.0e-12 -> fails the same tight threshold
    _write_track_csv(tmp_path / "sat_fail.csv", 5.0e-11)
    _write_track_csv(tmp_path / "phys_fail.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[
            _one_period(label="pass", sat="sat_pass.csv", phys="phys_pass.csv"),
            _one_period(label="fail", sat="sat_fail.csv", phys="phys_fail.csv"),
        ],
        threshold={"max": 1.0e-13},
    )

    assert output["per_period"]["pass"]["delta_+0h"]["passed"] is True
    assert output["per_period"]["fail"]["delta_+0h"]["passed"] is False
    assert output["passed"] is False


def _orbit_test_frame():
    # Two complete orbits (3 ascending + 3 descending hours each), bracketed by a leading
    # descending-only fragment and a trailing ascending-only fragment -- both incomplete, both
    # must be dropped rather than averaged in. Explicit "ascending" column (not derived).
    times = pd.date_range("2023-12-31 23:00:00", periods=14, freq="h")
    satellite_density = [999.0, 10, 20, 30, 40, 50, 60, 110, 120, 130, 140, 150, 160, 999.0]
    ascending = [False, True, True, True, False, False, False, True, True, True, False, False, False, True]
    return pd.DataFrame({
        "datetime": times, "lst": 12.0, "lat": 0.0, "alt_km": 400.0,
        "satellite_density": satellite_density, "physics_density": satellite_density,
        "rope_density": satellite_density, "ascending": ascending,
    })


def _orbit_test_frame_lat_derived():
    # Same two-complete-orbits-plus-leading-fragment shape as _orbit_test_frame(), but with no
    # "ascending" column at all -- matching the real satellite_track_csv schema in this project
    # (datetime,lst,lat,lon,alt_km,density), which never included one. Direction must be derived
    # from lat itself. No trailing fragment here: the very last row's direction is always a copy
    # of the second-to-last transition (no row after it to compare against), so an unambiguous
    # trailing-fragment case isn't constructible via pure lat-derivation -- covered instead by
    # _orbit_test_frame()'s explicit-column version above.
    times = pd.date_range("2024-01-01 00:00:00", periods=13, freq="h")
    satellite_density = [999.0, 10, 20, 30, 40, 50, 60, 110, 120, 130, 140, 150, 160]
    lat = [10, -80, -40, 0, 40, 0, -40, -80, -40, 0, 40, 0, -40]
    return pd.DataFrame({
        "datetime": times, "lst": 12.0, "lat": lat, "alt_km": 400.0,
        "satellite_density": satellite_density, "physics_density": satellite_density,
        "rope_density": satellite_density,
    })


def test_orbit_average_drops_incomplete_edge_fragments_and_averages_complete_orbits():
    result = _orbit_average(_orbit_test_frame(), "test period")

    assert len(result) == 2
    assert result["satellite_density"].tolist() == pytest.approx([35.0, 135.0])
    assert result["datetime"].iloc[0] == pd.Timestamp("2024-01-01 02:30:00")
    assert result["datetime"].iloc[1] == pd.Timestamp("2024-01-01 08:30:00")


def test_orbit_average_derives_direction_from_lat_when_ascending_column_absent():
    result = _orbit_average(_orbit_test_frame_lat_derived(), "test period")

    assert len(result) == 2
    assert result["satellite_density"].tolist() == pytest.approx([35.0, 135.0])
    assert result["datetime"].iloc[0] == pd.Timestamp("2024-01-01 03:30:00")
    assert result["datetime"].iloc[1] == pd.Timestamp("2024-01-01 09:30:00")


def test_orbit_average_missing_ascending_and_lat_raises():
    df = _orbit_test_frame().drop(columns=["ascending", "lat"])
    with pytest.raises(ValueError, match="ascending.*lat"):
        _orbit_average(df, "test period")


def test_satellite_orbit_density_orbit_averaged_end_to_end(tmp_path):
    # No "ascending" column -- matches the real satellite_track_csv schema; direction is derived
    # from lat.
    frame = _orbit_test_frame_lat_derived()
    frame[["datetime", "lst", "lat", "alt_km", "satellite_density"]].rename(
        columns={"satellite_density": "density"}
    ).to_csv(tmp_path / "sat.csv", index=False)
    frame[["datetime", "lst", "lat", "alt_km", "physics_density"]].rename(
        columns={"physics_density": "density"}
    ).to_csv(tmp_path / "phys.csv", index=False)
    fn = get_kind_function("satellite_orbit_density")

    period = _one_period(start="2023-12-31 23:00:00", end="2024-01-01 13:00:00")
    period["orbit_averaged"] = True

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period],
    )

    # RMSE against the FakeModel's constant 1.0e-12 output, computed over the 2 orbit-averaged
    # points (35, 135), not the 13 raw rows.
    assert output["per_period"]["p1"]["delta_+0h"]["value"] == pytest.approx(
        ((35.0 - 1e-12) ** 2 + (135.0 - 1e-12) ** 2) ** 0.5 / 2 ** 0.5, rel=1e-6,
    )
    assert (tmp_path / output["plots"][0]).is_file()


def test_satellite_orbit_density_uses_suite_labels(tmp_path, monkeypatch):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    captured = {}
    import rope_dev_tools.validation.checks.satellite_orbit_density as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], physics_model_label="WAM", rope_model_label="ROPE-WAM-V1",
        satellite_label="GRACE",
    )

    assert captured["series_keys"] == {"WAM", "ROPE-WAM-V1", "GRACE"}


def test_satellite_orbit_density_default_labels_unchanged(tmp_path, monkeypatch):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    captured = {}
    import rope_dev_tools.validation.checks.satellite_orbit_density as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()])

    assert captured["series_keys"] == {"physics_model", "rope_model", "satellite"}


def test_satellite_orbit_density_plot_satellite_data_false_omits_satellite_line(tmp_path, monkeypatch):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    captured = {}
    import rope_dev_tools.validation.checks.satellite_orbit_density as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    period = {**_one_period(), "plot_satellite_data": False}
    fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])

    assert captured["series_keys"] == {"physics_model", "rope_model"}


def test_satellite_orbit_density_plot_satellite_data_defaults_to_true(tmp_path):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()])

    comparison = pd.read_csv(tmp_path / output["data"][0])
    assert set(comparison["plot_satellite_data"]) == {True}


def test_replot_satellite_orbit_density_respects_plot_satellite_data_false(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "period": ["p1", "p1"], "datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "start_delta": [0, 0], "satellite_density": [1.0e-12, 1.1e-12],
        "physics_density": [0.9e-12, 1.0e-12], "rope_density": [1.0e-12, 1.05e-12],
        "orbit_averaged": [False, False], "plot_satellite_data": [False, False],
    })

    captured = {}
    import rope_dev_tools.validation.checks.satellite_orbit_density as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)
    replot_satellite_orbit_density({"validation_data/sat_test.csv": df}, id="sat_test", out_dir=tmp_path)

    assert captured["series_keys"] == {"physics_model", "rope_model"}


def test_replot_satellite_orbit_density_missing_column_defaults_to_true(tmp_path, monkeypatch):
    # CSVs written before this feature existed have no plot_satellite_data column at all.
    df = pd.DataFrame({
        "period": ["p1", "p1"], "datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "start_delta": [0, 0], "satellite_density": [1.0e-12, 1.1e-12],
        "physics_density": [0.9e-12, 1.0e-12], "rope_density": [1.0e-12, 1.05e-12],
    })

    captured = {}
    import rope_dev_tools.validation.checks.satellite_orbit_density as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)
    replot_satellite_orbit_density({"validation_data/sat_test.csv": df}, id="sat_test", out_dir=tmp_path)

    assert "satellite" in captured["series_keys"]


def _write_hourly_track_csv(path, density=1.0e-12, *, start="2024-01-01 00:00:00", n_hours=6):
    base = pd.Timestamp(start)
    lines = ["datetime,lst,lat,alt_km,density"]
    for h in range(n_hours):
        t = base + pd.Timedelta(hours=h)
        lines.append(f"{t},{12.0 + h},{h * 5.0},400.0,{density}")
    path.write_text("\n".join(lines) + "\n")


def test_satellite_orbit_density_default_start_delta_column_is_zero(tmp_path):
    _write_track_csv(tmp_path / "sat.csv", 1.0e-12)
    _write_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()],
    )

    comparison = pd.read_csv(tmp_path / output["data"][0])
    assert set(comparison["start_delta"]) == {0}


def test_satellite_orbit_density_start_delta_shifts_forecast_start(tmp_path):
    _write_hourly_track_csv(tmp_path / "sat.csv")
    _write_hourly_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")
    model = _FakeModel()

    fn(
        model, id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{**_one_period(end="2024-01-01 06:00:00"), "start_deltas": [-2, 0]}],
    )

    assert ("2023-12-31 22:00:00", "2024-01-01 06:00:00") in model.forecast_calls
    assert ("2024-01-01 00:00:00", "2024-01-01 06:00:00") in model.forecast_calls


def test_satellite_orbit_density_positive_start_delta_narrows_query_window(tmp_path):
    _write_hourly_track_csv(tmp_path / "sat.csv")
    _write_hourly_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    period = {**_one_period(end="2024-01-01 06:00:00"), "start_deltas": [0, 2]}
    output = fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])

    comparison = pd.read_csv(tmp_path / output["data"][0])
    delta0_times = set(comparison[comparison["start_delta"] == 0]["datetime"])
    delta2_times = set(comparison[comparison["start_delta"] == 2]["datetime"])
    assert len(delta0_times) == 6
    assert len(delta2_times) == 4


def test_satellite_orbit_density_start_delta_consuming_window_raises(tmp_path):
    _write_hourly_track_csv(tmp_path / "sat.csv")
    _write_hourly_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    period = {**_one_period(end="2024-01-01 06:00:00"), "start_deltas": [6]}
    with pytest.raises(ValueError, match="leaving no time"):
        fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])


def test_satellite_orbit_density_multiple_start_deltas_add_suffixed_rope_lines(tmp_path, monkeypatch):
    _write_hourly_track_csv(tmp_path / "sat.csv")
    _write_hourly_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    captured = {}
    import rope_dev_tools.validation.checks.satellite_orbit_density as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    period = {**_one_period(end="2024-01-01 06:00:00"), "start_deltas": [-2, 0]}
    fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period],
        physics_model_label="WAM", rope_model_label="ROPE-WAM-V1", satellite_label="GRACE",
    )

    assert captured["series_keys"] == {"WAM", "GRACE", "ROPE-WAM-V1 (Δ-2h)", "ROPE-WAM-V1 (Δ+0h)"}


def test_satellite_orbit_density_multiple_start_deltas_nested_output(tmp_path):
    _write_hourly_track_csv(tmp_path / "sat.csv")
    _write_hourly_track_csv(tmp_path / "phys.csv", 0.9e-12)
    fn = get_kind_function("satellite_orbit_density")

    period = {**_one_period(end="2024-01-01 06:00:00"), "start_deltas": [-2, 0]}
    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period],
        statistics=["bias"], threshold={"max": 1.0e-13},
    )

    assert set(output["per_period"]["p1"]) == {"delta_-2h", "delta_+0h"}
    assert set(output["statistics"]["p1"]) == {"delta_-2h", "delta_+0h"}
    assert output["passed"] is True  # both deltas' rope (constant 1.0e-12) match satellite exactly


def test_satellite_orbit_density_one_bad_delta_fails_whole_period(tmp_path):
    # rope is a constant 1.0e-12 regardless of delta; satellite is engineered so the delta=2
    # slice's mean differs sharply from rope while delta=0's doesn't, to prove the threshold
    # gate is AND'd across every delta, not just the first/primary one.
    rows = [
        ("2024-01-01 00:00:00", 12.0, 0.0, 400.0, 1.0e-12),
        ("2024-01-01 01:00:00", 12.0, 0.0, 400.0, 1.0e-12),
        ("2024-01-01 02:00:00", 12.0, 0.0, 400.0, 5.0e-11),
        ("2024-01-01 03:00:00", 12.0, 0.0, 400.0, 5.0e-11),
    ]
    _write_track_csv(tmp_path / "sat.csv", None, rows=rows)
    _write_track_csv(tmp_path / "phys.csv", None, rows=rows)
    fn = get_kind_function("satellite_orbit_density")

    period = {**_one_period(end="2024-01-01 04:00:00"), "start_deltas": [0, 2]}
    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period],
        threshold={"max": 1.0e-13},
    )

    assert output["per_period"]["p1"]["delta_+0h"]["passed"] is False  # includes the bad rows too
    assert output["per_period"]["p1"]["delta_+2h"]["passed"] is False
    assert output["passed"] is False


def test_satellite_orbit_density_orbit_averaged_per_delta(tmp_path):
    frame = _orbit_test_frame_lat_derived()
    frame[["datetime", "lst", "lat", "alt_km", "satellite_density"]].rename(
        columns={"satellite_density": "density"}
    ).to_csv(tmp_path / "sat.csv", index=False)
    frame[["datetime", "lst", "lat", "alt_km", "physics_density"]].rename(
        columns={"physics_density": "density"}
    ).to_csv(tmp_path / "phys.csv", index=False)
    fn = get_kind_function("satellite_orbit_density")

    period = _one_period(start="2023-12-31 23:00:00", end="2024-01-01 13:00:00")
    period["orbit_averaged"] = True
    period["start_deltas"] = [0]

    output = fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])

    assert output["per_period"]["p1"]["delta_+0h"]["value"] == pytest.approx(
        ((35.0 - 1e-12) ** 2 + (135.0 - 1e-12) ** 2) ** 0.5 / 2 ** 0.5, rel=1e-6,
    )


def test_replot_satellite_orbit_density_single_delta(tmp_path):
    df = pd.DataFrame({
        "period": ["p1", "p1"], "datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "start_delta": [0, 0], "satellite_density": [1.0e-12, 1.1e-12],
        "physics_density": [0.9e-12, 1.0e-12], "rope_density": [1.0e-12, 1.05e-12],
        "orbit_averaged": [False, False],
    })
    plots = replot_satellite_orbit_density({"validation_data/sat_test.csv": df}, id="sat_test", out_dir=tmp_path)
    assert len(plots) == 1
    assert (tmp_path / plots[0]).is_file()


def test_replot_satellite_orbit_density_multiple_deltas(tmp_path):
    df = pd.DataFrame({
        "period": ["p1"] * 4,
        "datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"] * 2,
        "start_delta": [-2, -2, 0, 0],
        "satellite_density": [1.0e-12, 1.1e-12, 1.0e-12, 1.1e-12],
        "physics_density": [0.9e-12, 1.0e-12, 0.9e-12, 1.0e-12],
        "rope_density": [0.95e-12, 1.0e-12, 1.0e-12, 1.05e-12],
        "orbit_averaged": [False, False, False, False],
    })
    plots = replot_satellite_orbit_density({"validation_data/sat_test.csv": df}, id="sat_test", out_dir=tmp_path)
    assert len(plots) == 1
    assert (tmp_path / plots[0]).is_file()


def test_satellite_orbit_density_rope_vs_physics_model_uses_physics_not_satellite(tmp_path):
    # rope (1.0e-12 via _FakeModel) exactly matches physics (also 1.0e-12) but differs from
    # satellite (0.9e-12) -- rope_vs_physics_model must reflect the rope/physics comparison
    # specifically, not silently reuse rope_vs_satellite's numbers.
    _write_track_csv(tmp_path / "sat.csv", 0.9e-12)
    _write_track_csv(tmp_path / "phys.csv", 1.0e-12)
    fn = get_kind_function("satellite_orbit_density")

    output = fn(
        _FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[_one_period()], statistics=["bias"],
    )

    stats = output["statistics"]["p1"]["delta_+0h"]
    assert stats["rope_vs_physics_model"]["bias"] == pytest.approx(0.0, abs=1e-20)
    assert stats["rope_vs_satellite"]["bias"] == pytest.approx(1.0e-13, abs=1e-20)


def test_satellite_orbit_density_stats_text_includes_both_comparisons(tmp_path, monkeypatch):
    _write_track_csv(tmp_path / "sat.csv", 0.9e-12)
    _write_track_csv(tmp_path / "phys.csv", 1.0e-12)
    fn = get_kind_function("satellite_orbit_density")

    captured = {}
    import rope_dev_tools.validation.checks.satellite_orbit_density as mod

    def fake_line_plot(panels, **kwargs):
        captured["stats_text"] = panels[0]["stats_text"]
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    fn(_FakeModel(), id="sat_test", out_dir=tmp_path, suite_dir=tmp_path,
       periods=[_one_period()], statistics=["bias"])

    assert "rope_vs_satellite" in captured["stats_text"]
    assert "rope_vs_physics" in captured["stats_text"]
