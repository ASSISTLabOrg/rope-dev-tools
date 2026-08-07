"""Plotting primitives: synthetic arrays in, PNG/GIF files out."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageSequence  # noqa: E402

from rope_dev_tools.validation.plots import (
    doy_lat_plot,
    harmonic_fft_plot,
    line_plot,
    lonlat_animation,
    lonlat_plot,
)


def test_line_plot_writes_file_with_stats_text(tmp_path):
    out_path = tmp_path / "line.png"
    x = np.arange(5)
    panels = [{"title": "400 km", "ylabel": "density", "series": {"truth": (x, x), "model": (x, x + 1)},
               "stats_text": "log_bias: 1.2e-14"}]
    line_plot(panels, out_path=out_path)
    assert out_path.is_file()


def test_line_plot_shades_uncertainty_band_when_a_series_has_a_third_element(tmp_path):
    out_path = tmp_path / "line.png"
    x = np.arange(5)
    y = x.astype(float)
    panels = [{"title": "400 km", "ylabel": "density",
               "series": {"model": (x, y, np.full_like(y, 0.5)), "truth": (x, y)}}]
    line_plot(panels, out_path=out_path)
    assert out_path.is_file()


def test_line_plot_forwards_plot_kwargs(tmp_path):
    out_path = tmp_path / "line.png"
    x = np.arange(3)
    panels = [{"title": "t", "ylabel": "y", "series": {"a": (x, x)}}]
    line_plot(panels, out_path=out_path, plot_kwargs={"linewidth": 3})
    assert out_path.is_file()


def test_line_plot_handles_long_datetime_series(tmp_path):
    out_path = tmp_path / "line.png"
    x = np.array(["2023-01-01T00:00:00"], dtype="datetime64[ns]") + np.arange(30 * 24 * 6) * np.timedelta64(10, "m")
    y = np.sin(np.arange(len(x)) * 0.01)
    panels = [{"title": "long period", "ylabel": "density", "series": {"satellite": (x, y)},
               "stats_text": "rmse: 1.0e-13"}]
    line_plot(panels, out_path=out_path)
    assert out_path.is_file()


def test_line_plot_does_not_downsample_long_series(tmp_path, monkeypatch):
    """No automatic time-based thinning -- every raw point is plotted (see satellite_orbit_density's
    orbit_averaged for the explicit, opt-in reduction that replaced the old auto-downsampling)."""
    import matplotlib.axes

    plotted_lengths = []
    real_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        if args and hasattr(args[0], "__len__"):
            plotted_lengths.append(len(args[0]))
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    out_path = tmp_path / "line.png"
    x = np.array(["2023-01-01T00:00:00"], dtype="datetime64[ns]") + np.arange(30 * 24 * 6) * np.timedelta64(10, "m")
    y = np.sin(np.arange(len(x)) * 0.01)
    panels = [{"title": "long period", "ylabel": "density", "series": {"satellite": (x, y)}}]
    line_plot(panels, out_path=out_path)

    assert plotted_lengths == [len(x)]


def test_line_plot_adds_gridlines(tmp_path, monkeypatch):
    import matplotlib.axes

    grid_calls = []
    real_grid = matplotlib.axes.Axes.grid

    def spy_grid(self, *args, **kwargs):
        grid_calls.append(args)
        return real_grid(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "grid", spy_grid)

    out_path = tmp_path / "line.png"
    x = np.arange(5)
    panels = [{"title": "t", "ylabel": "y", "series": {"a": (x, x)}}]
    line_plot(panels, out_path=out_path)

    # Axes.__clear() calls self.grid(False) (and a second self.grid(...) from rcParams) during
    # axis init, before line_plot's own explicit call -- only the last call reflects what's
    # actually drawn.
    assert grid_calls[-1][0] is True


def test_line_plot_uses_doy_labels_for_spans_over_a_day(tmp_path, monkeypatch):
    import datetime

    import matplotlib.axis
    import matplotlib.dates as mdates

    formatter_calls = []
    real_set_formatter = matplotlib.axis.XAxis.set_major_formatter

    def spy_set_formatter(self, formatter, *args, **kwargs):
        formatter_calls.append(formatter)
        return real_set_formatter(self, formatter, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axis.XAxis, "set_major_formatter", spy_set_formatter)

    out_path = tmp_path / "line.png"
    x = np.array(["2023-01-01T00:00:00"], dtype="datetime64[ns]") + np.arange(30) * np.timedelta64(1, "D")
    y = np.arange(30)
    panels = [{"title": "long", "ylabel": "y", "series": {"a": (x, y)}}]
    line_plot(panels, out_path=out_path)

    # matplotlib's own axis setup installs formatters of its own (a default ScalarFormatter at
    # axis init, then an AutoDateFormatter once it sees datetime data) before line_plot's explicit
    # override -- only the last call is the one that's actually used at render time.
    jan_15 = mdates.date2num(datetime.datetime(2023, 1, 15))
    assert formatter_calls[-1](jan_15) == "15"


def test_line_plot_doy_labels_include_year_across_year_boundary(tmp_path, monkeypatch):
    import datetime

    import matplotlib.axis
    import matplotlib.dates as mdates

    formatter_calls = []
    real_set_formatter = matplotlib.axis.XAxis.set_major_formatter

    def spy_set_formatter(self, formatter, *args, **kwargs):
        formatter_calls.append(formatter)
        return real_set_formatter(self, formatter, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axis.XAxis, "set_major_formatter", spy_set_formatter)

    out_path = tmp_path / "line.png"
    x = np.array(["2022-12-20T00:00:00"], dtype="datetime64[ns]") + np.arange(20) * np.timedelta64(1, "D")
    y = np.arange(20)
    panels = [{"title": "cross-year", "ylabel": "y", "series": {"a": (x, y)}}]
    line_plot(panels, out_path=out_path)

    dec_25 = mdates.date2num(datetime.datetime(2022, 12, 25))
    assert formatter_calls[-1](dec_25) == "2022-359"


def test_line_plot_keeps_default_date_labels_for_short_spans(tmp_path, monkeypatch):
    import matplotlib.axis

    formatter_calls = []
    real_set_formatter = matplotlib.axis.XAxis.set_major_formatter

    def spy_set_formatter(self, formatter, *args, **kwargs):
        formatter_calls.append(formatter)
        return real_set_formatter(self, formatter, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axis.XAxis, "set_major_formatter", spy_set_formatter)

    out_path = tmp_path / "line.png"
    x = np.array(["2023-01-01T00:00:00"], dtype="datetime64[ns]") + np.arange(3) * np.timedelta64(1, "h")
    y = np.arange(3)
    panels = [{"title": "short", "ylabel": "y", "series": {"a": (x, y)}}]
    line_plot(panels, out_path=out_path)

    # matplotlib's own axis setup still installs its default formatters (ScalarFormatter, then
    # AutoDateFormatter once it sees datetime data) -- line_plot must not add its own DOY
    # override on top for a short span, so the last one standing is matplotlib's, not a
    # FuncFormatter.
    import matplotlib.ticker

    assert not isinstance(formatter_calls[-1], matplotlib.ticker.FuncFormatter)


def test_line_plot_suppresses_suptitle_for_a_single_panel(tmp_path, monkeypatch):
    import matplotlib.figure

    suptitle_calls = []
    real_suptitle = matplotlib.figure.Figure.suptitle

    def spy_suptitle(self, *args, **kwargs):
        suptitle_calls.append(args)
        return real_suptitle(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", spy_suptitle)

    out_path = tmp_path / "line.png"
    x = np.arange(3)
    panels = [{"title": "March 2015 Storm", "ylabel": "y", "series": {"a": (x, x)}}]
    line_plot(panels, out_path=out_path, suptitle="Satellite Track Density — March 2015 Storm")

    # a single panel already carries this exact identifying text via its own ax.set_title -- a
    # figure-level suptitle on top of that would just repeat it a second time.
    assert suptitle_calls == []


def test_line_plot_keeps_suptitle_for_multiple_panels(tmp_path, monkeypatch):
    import matplotlib.figure

    suptitle_calls = []
    real_suptitle = matplotlib.figure.Figure.suptitle

    def spy_suptitle(self, *args, **kwargs):
        suptitle_calls.append(args)
        return real_suptitle(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", spy_suptitle)

    out_path = tmp_path / "line.png"
    x = np.arange(3)
    panels = [
        {"title": "400 km", "ylabel": "y", "series": {"a": (x, x)}},
        {"title": "300 km", "ylabel": "y", "series": {"a": (x, x)}},
    ]
    line_plot(panels, out_path=out_path, suptitle="Average Density — March 2015")

    # multiple panels don't individually carry the check id/period -- the suptitle is the only
    # place that context appears, so it must stay.
    assert suptitle_calls == [("Average Density — March 2015",)]


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


def test_lonlat_animation_color_scale_is_constant_across_frames(tmp_path):
    """Regression test: Pillow's GIF writer computes an independent 256-color palette per frame
    by default, which visibly shifts a fixed data value's rendered color over the course of an
    animation even though vmin/vmax/cmap never change. Embeds a fixed probe value (4.5) in every
    frame's array center alongside surrounding data that grows each frame, and checks the probe's
    rendered color doesn't drift. Sampled at the image center -- for a single n_cols=1 panel this
    reliably lands inside the plotted data area regardless of exact axes margins."""
    n_frames = 8
    frames = []
    for i in range(n_frames):
        arr = np.linspace(0.0, max(float(i), 0.1), 8 * 6).reshape(8, 6)
        arr[4, 3] = 4.5  # fixed probe value, present in every frame
        frames.append(arr)

    out_path = tmp_path / "anim.gif"
    lonlat_animation(
        [{"title": "a", "frames": frames}], timestamps=[f"t{i}" for i in range(n_frames)],
        n_rows=1, n_cols=1, lat_range=(-80.0, 80.0), out_path=out_path, vmin=0.0, vmax=9.0,
    )

    im = Image.open(out_path)
    rgb_frames = [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(im)]
    assert len(rgb_frames) == n_frames

    w, h = rgb_frames[0].size
    probe_xy = (w // 2, h // 2)
    probe_colors = {frame.getpixel(probe_xy) for frame in rgb_frames}
    assert probe_colors != {(255, 255, 255)}, "probe landed on background, not plot data"
    assert len(probe_colors) == 1, f"probe pixel color drifted across frames: {probe_colors}"

    # arr[0, 0] (lon index 0, lat index 0) sits at the bottom-left of the plotted data, imshow's
    # origin="lower" + the array transpose in lonlat_animation together place it a few pixels in
    # from the axes' bottom-left corner.
    w, h = rgb_frames[0].size
    probe_xy = (int(w * 0.09), int(h * 0.72))
    probe_colors = {frame.getpixel(probe_xy) for frame in rgb_frames}
    assert len(probe_colors) == 1, f"probe pixel color drifted across frames: {probe_colors}"


def _stats_animation_inputs(n_frames=4):
    rng = np.random.default_rng(0)
    frames_a = [rng.random((8, 6)) for _ in range(n_frames)]
    frames_b = [rng.random((8, 6)) for _ in range(n_frames)]
    timestamps = [f"2023-01-01 {h:02d}:00:00" for h in range(n_frames)]
    return frames_a, frames_b, timestamps


def test_lonlat_animation_stats_series_adds_extra_panel(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        captured["args"] = args
        return real_subplots(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    frames_a, frames_b, timestamps = _stats_animation_inputs()
    out_path = tmp_path / "anim.gif"
    lonlat_animation(
        [{"title": "physics", "frames": frames_a}, {"title": "rope", "frames": frames_b}],
        timestamps=timestamps, n_rows=1, n_cols=2, lat_range=(-80.0, 80.0), out_path=out_path,
        stats_series={"log_bias": [0.1, 0.2, 0.15, 0.05]},
    )

    assert captured["args"][:2] == (1, 3)  # 2 heatmap panels + 1 stats panel
    assert out_path.is_file()


def test_lonlat_animation_stats_uncertainty_series_shades_a_band(tmp_path):
    frames_a, frames_b, timestamps = _stats_animation_inputs()
    out_path = tmp_path / "anim.gif"
    lonlat_animation(
        [{"title": "physics", "frames": frames_a}, {"title": "rope", "frames": frames_b}],
        timestamps=timestamps, n_rows=1, n_cols=2, lat_range=(-80.0, 80.0), out_path=out_path,
        stats_series={"log_bias": [0.1, 0.2, 0.15, 0.05]},
        stats_uncertainty_series={"log_bias": [0.02, 0.03, 0.02, 0.01]},
    )
    assert out_path.is_file()


def test_lonlat_animation_without_stats_series_keeps_original_panel_count(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        captured["args"] = args
        return real_subplots(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    frames_a, frames_b, timestamps = _stats_animation_inputs()
    out_path = tmp_path / "anim.gif"
    lonlat_animation(
        [{"title": "physics", "frames": frames_a}, {"title": "rope", "frames": frames_b}],
        timestamps=timestamps, n_rows=1, n_cols=2, lat_range=(-80.0, 80.0), out_path=out_path,
    )

    assert captured["args"][:2] == (1, 2)  # unchanged: no stats panel added


def test_lonlat_animation_stats_line_grows_point_by_point(tmp_path, monkeypatch):
    import matplotlib.axes

    set_data_lengths = []
    real_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        lines = real_plot(self, *args, **kwargs)
        for line in lines:
            real_set_data = line.set_data

            def spy_set_data(x, y, _real=real_set_data):
                set_data_lengths.append(len(np.asarray(x)))
                return _real(x, y)

            line.set_data = spy_set_data
        return lines

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    frames_a, frames_b, timestamps = _stats_animation_inputs(n_frames=4)
    out_path = tmp_path / "anim.gif"
    lonlat_animation(
        [{"title": "physics", "frames": frames_a}, {"title": "rope", "frames": frames_b}],
        timestamps=timestamps, n_rows=1, n_cols=2, lat_range=(-80.0, 80.0), out_path=out_path,
        stats_series={"log_bias": [0.1, 0.2, 0.15, 0.05]},
    )

    # ax.plot([], [], ...)'s initial empty data is set via the Line2D constructor directly, not a
    # separate set_data() call our spy would catch -- what we do catch is exactly the per-frame
    # growth, one more point revealed each time.
    assert set_data_lengths == [1, 2, 3, 4]


def test_lonlat_animation_multiple_stats_series_all_present(tmp_path):
    frames_a, frames_b, timestamps = _stats_animation_inputs(n_frames=3)
    out_path = tmp_path / "anim.gif"
    lonlat_animation(
        [{"title": "physics", "frames": frames_a}, {"title": "rope", "frames": frames_b}],
        timestamps=timestamps, n_rows=1, n_cols=2, lat_range=(-80.0, 80.0), out_path=out_path,
        stats_series={"log_bias": [0.1, 0.2, 0.3], "abs_log_bias": [0.1, 0.2, 0.1], "log_rmse": [0.2, 0.3, 0.25]},
    )
    assert out_path.is_file()


def test_doy_lat_plot_writes_file(tmp_path):
    out_path = tmp_path / "doy_lat.png"
    grid = np.random.default_rng(0).random((10, 5))
    doy_lat_plot(grid, doy_edges=np.arange(11), lat_edges=np.linspace(-50, 50, 6), out_path=out_path)
    assert out_path.is_file()


def test_harmonic_fft_plot_writes_file(tmp_path):
    out_path = tmp_path / "fft.png"
    freqs = np.linspace(0.01, 0.5, 20)
    magnitude = np.abs(np.sin(freqs * 50)) + 0.1
    panels = [
        {"title": "WAM", "series": {"150km": (freqs, magnitude), "300km": (freqs, magnitude * 2)}},
        {"title": "ROPE-WAM", "series": {"150km": (freqs, magnitude), "300km": (freqs, magnitude * 2)}},
    ]
    harmonic_fft_plot(panels, harmonic_freqs_per_hour=[1 / 24, 1 / 12, 1 / 8, 1 / 6], out_path=out_path)
    assert out_path.is_file()


def test_harmonic_fft_plot_uses_log_yscale(tmp_path, monkeypatch):
    import matplotlib.axes

    captured = []
    real_set_yscale = matplotlib.axes.Axes.set_yscale

    def spy_set_yscale(self, value, *args, **kwargs):
        captured.append(value)
        return real_set_yscale(self, value, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_yscale", spy_set_yscale)

    out_path = tmp_path / "fft.png"
    freqs = np.linspace(0.01, 0.5, 10)
    magnitude = np.ones_like(freqs)
    panels = [{"title": "WAM", "series": {"150km": (freqs, magnitude)}}]
    harmonic_fft_plot(panels, harmonic_freqs_per_hour=[1 / 24], out_path=out_path)

    assert captured == ["log"]


def test_harmonic_fft_plot_draws_vertical_line_per_harmonic(tmp_path, monkeypatch):
    import matplotlib.axes

    axvline_calls = []
    real_axvline = matplotlib.axes.Axes.axvline

    def spy_axvline(self, x=0, *args, **kwargs):
        axvline_calls.append(x)
        return real_axvline(self, x, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axvline", spy_axvline)

    out_path = tmp_path / "fft.png"
    freqs = np.linspace(0.01, 0.5, 10)
    magnitude = np.ones_like(freqs)
    panels = [{"title": "WAM", "series": {"150km": (freqs, magnitude)}}]
    harmonics = [1 / 24, 1 / 12, 1 / 8, 1 / 6]
    harmonic_fft_plot(panels, harmonic_freqs_per_hour=harmonics, out_path=out_path)

    assert axvline_calls == harmonics
