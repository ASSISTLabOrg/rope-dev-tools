"""Plotting primitives: synthetic arrays in, PNG/GIF files out."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.plots import doy_lat_plot, line_plot, lonlat_animation, lonlat_plot


def test_line_plot_writes_file_with_stats_text(tmp_path):
    out_path = tmp_path / "line.png"
    x = np.arange(5)
    panels = [{"title": "400 km", "ylabel": "density", "series": {"truth": (x, x), "model": (x, x + 1)},
               "stats_text": "log_bias: 1.2e-14"}]
    line_plot(panels, out_path=out_path)
    assert out_path.is_file()


def test_line_plot_forwards_plot_kwargs(tmp_path):
    out_path = tmp_path / "line.png"
    x = np.arange(3)
    panels = [{"title": "t", "ylabel": "y", "series": {"a": (x, x)}}]
    line_plot(panels, out_path=out_path, plot_kwargs={"linewidth": 3})
    assert out_path.is_file()


def test_lonlat_plot_writes_file(tmp_path):
    out_path = tmp_path / "lonlat.png"
    grid = np.random.default_rng(0).random((8, 6))
    panels = [{"title": "00:00", "grid": grid}, {"title": "12:00", "grid": grid}]
    lonlat_plot(panels, n_rows=1, n_cols=2, lat_range=(-80.0, 80.0), out_path=out_path)
    assert out_path.is_file()


def test_lonlat_plot_forwards_imshow_kwargs(tmp_path):
    out_path = tmp_path / "lonlat.png"
    grid = np.random.default_rng(0).random((8, 6))
    lonlat_plot([{"title": "t", "grid": grid}], n_rows=1, n_cols=1, lat_range=(-80.0, 80.0),
                out_path=out_path, imshow_kwargs={"alpha": 0.5})
    assert out_path.is_file()


def test_lonlat_animation_writes_file(tmp_path):
    out_path = tmp_path / "anim.gif"
    rng = np.random.default_rng(0)
    frames_a = [rng.random((8, 6)) for _ in range(3)]
    frames_b = [rng.random((8, 6)) for _ in range(3)]
    lonlat_animation(
        [{"title": "physics", "frames": frames_a}, {"title": "rope", "frames": frames_b}],
        timestamps=["t0", "t1", "t2"], n_rows=1, n_cols=2, lat_range=(-80.0, 80.0), out_path=out_path,
    )
    assert out_path.is_file()


def test_doy_lat_plot_writes_file(tmp_path):
    out_path = tmp_path / "doy_lat.png"
    grid = np.random.default_rng(0).random((10, 5))
    doy_lat_plot(grid, doy_edges=np.arange(11), lat_edges=np.linspace(-50, 50, 6), out_path=out_path)
    assert out_path.is_file()
