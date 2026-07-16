"""avg_density_vs_time — one kind, any number/length of periods, optional statistics and backend gate."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function


class _FakeModel:
    backend_name = "wrapper"

    def forecast(self, start, end):
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


def test_avg_density_vs_time_computes_requested_statistics(tmp_path):
    _write_truth_csv(tmp_path / "truth.csv")
    fn = get_kind_function("avg_density_vs_time")

    output = fn(
        _FakeModel(), id="avg_test", out_dir=tmp_path, suite_dir=tmp_path,
        periods=[{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 01:00:00",
                  "physics_avg_csv": "truth.csv"}],
        altitudes_km=[400.0], statistics=["bias"],
    )

    assert output["statistics"]["p1"]["400.0km"]["model_vs_truth"].keys() == {"bias"}


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
