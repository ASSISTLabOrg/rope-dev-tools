"""avg_density_vs_time — one kind, any number/length of periods, optional statistics and backend gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function
from rope_dev_tools.validation.checks.avg_density_vs_time import replot_avg_density_vs_time


class _FakeModel:
    backend_name = "wrapper"

    def __init__(self):
        self.forecast_calls = []

    def forecast(self, start, end):
        self.forecast_calls.append((start, end))
        return {"window_start": start, "window_end": end}

    def query_grid(self, time, alt_km):
        return np.full((8, 6), 1.0e-12)


def _write_truth_csv(path):
    path.write_text(
        "datetime,alt_km,density\n"
        "2024-01-01 00:00:00,400.0,1.0e-12\n"
        "2024-01-01 01:00:00,400.0,1.1e-12\n"
    )


def test_avg_density_vs_time_writes_plot_and_data(tmp_path):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                  "physics_avg_csv": "truth.csv"}],
        altitudes_km=[400.0],
    )

    assert (tmp_path / output["plots"][0]).is_file()
    assert (tmp_path / output["data"][0]).is_file()
    assert "statistics" not in output


def test_avg_density_vs_time_uses_suite_labels(tmp_path, monkeypatch):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    captured = {}
    import rope_dev_tools.validation.checks.avg_density_vs_time as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                  "physics_avg_csv": "truth.csv"}],
        altitudes_km=[400.0], physics_model_label="WAM", rope_model_label="ROPE-WAM-V1",
    )

    assert captured["series_keys"] == {"WAM", "ROPE-WAM-V1"}


def test_avg_density_vs_time_default_labels_unchanged(tmp_path, monkeypatch):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    captured = {}
    import rope_dev_tools.validation.checks.avg_density_vs_time as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                  "physics_avg_csv": "truth.csv"}],
        altitudes_km=[400.0],
    )

    assert captured["series_keys"] == {"truth", "model"}


def test_avg_density_vs_time_computes_requested_statistics(tmp_path):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                  "physics_avg_csv": "truth.csv"}],
        altitudes_km=[400.0], statistics=["bias"],
    )

    # statistics are always nested one level deeper by start_delta, even for the default [0]-only
    # case -- a uniform output shape regardless of whether start_deltas is actually used.
    assert output["statistics"]["p1"]["400.0km"]["delta_+0h"]["model_vs_truth"].keys() == {"bias"}


def test_avg_density_vs_time_missing_altitude_raises(tmp_path):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                      "physics_avg_csv": "truth.csv"}],
            altitudes_km=[900.0],
        )


def test_avg_density_vs_time_requires_exported_model_gate(tmp_path):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                      "physics_avg_csv": "truth.csv"}],
            altitudes_km=[400.0], requires_exported_model=True,
        )


def test_avg_density_vs_time_filters_rows_outside_period(tmp_path):
    (tmp_path / "truth.csv").write_text(
        "datetime,alt_km,density\n"
        "2023-12-31 23:00:00,400.0,9.0e-12\n"  # outside [start, end) -- must be excluded
        "2024-01-01 00:00:00,400.0,1.0e-12\n"
        "2024-01-01 01:00:00,400.0,1.1e-12\n"
        "2024-01-02 00:00:00,400.0,9.0e-12\n"  # outside [start, end) -- must be excluded
    )
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 02:00:00",
                  "physics_avg_csv": "truth.csv"}],
        altitudes_km=[400.0],
    )

    comparison = pd.read_csv(tmp_path / output["data"][0])
    assert set(comparison["truth_density"]) == {1.0e-12, 1.1e-12}
    assert len(comparison) == 2


def test_avg_density_vs_time_period_with_no_rows_in_window_raises(tmp_path):
    (tmp_path / "truth.csv").write_text(
        "datetime,alt_km,density\n"
        "2020-01-01 00:00:00,400.0,1.0e-12\n"
    )
    fn = get_kind_function("avg_density_vs_time")

    with pytest.raises(ValueError):
        fn(
            _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 02:00:00",
                      "physics_avg_csv": "truth.csv"}],
            altitudes_km=[400.0],
        )


def test_avg_density_vs_time_multi_file_physics_avg_csv(tmp_path):
    (tmp_path / "truth_a.csv").write_text(
        "datetime,alt_km,density\n"
        "2024-01-01 00:00:00,400.0,1.0e-12\n"
    )
    (tmp_path / "truth_b.csv").write_text(
        "datetime,alt_km,density\n"
        "2024-01-01 01:00:00,400.0,1.1e-12\n"
    )
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 02:00:00",
                  "physics_avg_csv": ["truth_a.csv", "truth_b.csv"]}],
        altitudes_km=[400.0],
    )

    comparison = pd.read_csv(tmp_path / output["data"][0])
    assert len(comparison) == 2
    assert set(comparison["truth_density"]) == {1.0e-12, 1.1e-12}


def _write_hourly_truth_csv(path, *, start="2024-01-01 00:00:00", n_hours=6):
    lines = ["datetime,alt_km,density"]
    base = pd.Timestamp(start)
    for h in range(n_hours):
        t = base + pd.Timedelta(hours=h)
        lines.append(f"{t},400.0,{1.0e-12 + h * 1.0e-13}")
    path.write_text("\n".join(lines) + "\n")


def test_avg_density_vs_time_default_start_delta_column_is_zero(tmp_path):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                  "physics_avg_csv": "truth.csv"}],
        altitudes_km=[400.0],
    )

    comparison = pd.read_csv(tmp_path / output["data"][0])
    assert set(comparison["start_delta"]) == {0}


def test_avg_density_vs_time_start_delta_shifts_forecast_start(tmp_path):
    _write_hourly_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")
    model = _FakeModel()

    fn(
        model, id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
                  "physics_avg_csv": "truth.csv", "start_deltas": [-2, 0]}],
        altitudes_km=[400.0],
    )

    assert ("2023-12-31 22:00:00", "2024-01-01 06:00:00") in model.forecast_calls
    assert ("2024-01-01 00:00:00", "2024-01-01 06:00:00") in model.forecast_calls


def test_avg_density_vs_time_positive_start_delta_narrows_query_window(tmp_path):
    _write_hourly_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
                  "physics_avg_csv": "truth.csv", "start_deltas": [0, 2]}],
        altitudes_km=[400.0],
    )

    comparison = pd.read_csv(tmp_path / output["data"][0])
    delta0_times = set(comparison[comparison["start_delta"] == 0]["datetime"])
    delta2_times = set(comparison[comparison["start_delta"] == 2]["datetime"])
    assert delta0_times == {
        "2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00",
        "2024-01-01 03:00:00", "2024-01-01 04:00:00", "2024-01-01 05:00:00",
    }
    assert delta2_times == {
        "2024-01-01 02:00:00", "2024-01-01 03:00:00", "2024-01-01 04:00:00", "2024-01-01 05:00:00",
    }


def test_avg_density_vs_time_start_delta_consuming_window_raises(tmp_path):
    _write_hourly_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    with pytest.raises(ValueError, match="leaving no time"):
        fn(
            _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
            periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
                      "physics_avg_csv": "truth.csv", "start_deltas": [6]}],
            altitudes_km=[400.0],
        )


def test_avg_density_vs_time_multiple_start_deltas_add_suffixed_rope_lines(tmp_path, monkeypatch):
    _write_hourly_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    captured = {}
    import rope_dev_tools.validation.checks.avg_density_vs_time as mod

    def fake_line_plot(panels, **kwargs):
        captured["series_keys"] = set(panels[0]["series"])
        return "plots/fake.png"

    monkeypatch.setattr(mod, "line_plot", fake_line_plot)

    fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
                  "physics_avg_csv": "truth.csv", "start_deltas": [-2, 0]}],
        altitudes_km=[400.0], physics_model_label="WAM", rope_model_label="ROPE-WAM-V1",
    )

    assert captured["series_keys"] == {"WAM", "ROPE-WAM-V1 (Δ-2h)", "ROPE-WAM-V1 (Δ+0h)"}


def test_avg_density_vs_time_multiple_start_deltas_nested_statistics(tmp_path):
    _write_hourly_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
                  "physics_avg_csv": "truth.csv", "start_deltas": [-2, 0]}],
        altitudes_km=[400.0], statistics=["bias"],
    )

    per_alt = output["statistics"]["p1"]["400.0km"]
    assert set(per_alt) == {"delta_-2h", "delta_+0h"}
    assert per_alt["delta_-2h"]["model_vs_truth"].keys() == {"bias"}


def test_replot_avg_density_vs_time_single_delta(tmp_path):
    df = pd.DataFrame({
        "period": ["p1", "p1"], "datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "alt_km": [400.0, 400.0], "start_delta": [0, 0],
        "truth_density": [1.0e-12, 1.1e-12], "model_density": [1.0e-12, 1.05e-12],
    })
    plots = replot_avg_density_vs_time({"validation_data/avg_test.csv": df}, id="avg_test", out_dir=tmp_path)
    assert len(plots) == 1
    assert (tmp_path / plots[0]).is_file()


def test_replot_avg_density_vs_time_multiple_deltas(tmp_path):
    df = pd.DataFrame({
        "period": ["p1"] * 4,
        "datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"] * 2,
        "alt_km": [400.0] * 4, "start_delta": [-2, -2, 0, 0],
        "truth_density": [1.0e-12, 1.1e-12, 1.0e-12, 1.1e-12],
        "model_density": [0.9e-12, 1.0e-12, 1.0e-12, 1.05e-12],
    })
    plots = replot_avg_density_vs_time({"validation_data/avg_test.csv": df}, id="avg_test", out_dir=tmp_path)
    assert len(plots) == 1
    assert (tmp_path / plots[0]).is_file()
