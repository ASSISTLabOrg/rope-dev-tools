"""harmonic_fft — FFT of density time series at fixed spatial points, physics vs rope, no error metrics."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from rope_dev_tools.validation.checks import get_kind_function
from rope_dev_tools.validation.checks.harmonic_fft import _fft_magnitude, _grid_index, replot_harmonic_fft

_LON_VALUES = np.array([0.0, 90.0, 180.0, 270.0])
_LAT_VALUES = np.array([-80.0, -40.0, 0.0, 40.0, 80.0])
_ALTITUDES = np.array([150.0, 300.0, 450.0])


class _FakeModel:
    grid = {"lat_min_deg": -80.0, "lat_max_deg": 80.0}

    def __init__(self):
        self.forecast_calls = []
        self.query_calls = []

    def forecast(self, start, end):
        self.forecast_calls.append((start, end))
        return {"window_start": start, "window_end": end}

    def query(self, time, lst, lat, alt_km):
        self.query_calls.append((time, lst, lat, alt_km))
        hour = int(time.split(" ")[1].split(":")[0])
        # a real diurnal wobble, not a flat constant -- a flat signal de-biases to all zeros,
        # which can't be log-scaled (harmless in practice, just noisy in test output).
        density = 1.0e-12 * (1.0 + 0.1 * np.sin(2 * np.pi * hour / 24.0))
        return {"density": density, "uncertainty": 0.0}


def _write_physics_npz(path, n_hours=48):
    base = datetime(2024, 1, 1)
    times = [(base + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") for h in range(n_hours)]
    n_lon, n_lat, n_alt = len(_LON_VALUES), len(_LAT_VALUES), len(_ALTITUDES)
    hours = np.arange(n_hours)
    wobble = 1.0 + 0.1 * np.sin(2 * np.pi * hours / 24.0)
    density = wobble[:, None, None, None] * np.full((n_hours, n_alt, n_lon, n_lat), 1.0e-12)
    np.savez(
        path,
        times=np.array(times), lon_values=_LON_VALUES, n_lat=n_lat,
        lat_min_deg=float(_LAT_VALUES.min()), lat_max_deg=float(_LAT_VALUES.max()),
        altitudes_km=_ALTITUDES, density=density,
    )


def _one_period(label="p1", start="2024-01-01 00:00:00", end="2024-01-03 00:00:00",
                npz="phys.npz", lon_deg=0.0, lat_deg=0.0, altitudes_km=(150.0, 300.0),
                alt_km=300.0, lats_deg=(-40.0, 0.0, 40.0)):
    return {
        "label": label, "start": start, "end": end, "physics_model_hourly_npz": npz,
        "lon_deg": lon_deg, "lat_deg": lat_deg, "altitudes_km": list(altitudes_km),
        "alt_km": alt_km, "lats_deg": list(lats_deg),
    }


def test_fft_magnitude_peak_near_diurnal_frequency():
    hours = np.arange(96)  # 4 full days, hourly
    signal = 10.0 + 3.0 * np.sin(2 * np.pi * hours / 24.0)
    freqs, magnitude = _fft_magnitude(signal, cutoff=0.5)
    peak_freq = freqs[np.argmax(magnitude)]
    assert peak_freq == pytest.approx(1.0 / 24.0, abs=0.005)


def test_fft_magnitude_excludes_dc_and_above_cutoff():
    signal = np.ones(48) * 5.0  # pure DC -- de-biased, spectrum should be ~0 everywhere kept
    freqs, magnitude = _fft_magnitude(signal, cutoff=0.5)
    assert np.all(freqs > 0)
    assert np.all(freqs <= 0.5)
    assert np.allclose(magnitude, 0.0, atol=1e-9)


def test_fft_magnitude_respects_lower_cutoff():
    hours = np.arange(48)
    signal = np.sin(2 * np.pi * hours / 3.0)  # fast oscillation, period 3h -> freq 1/3
    freqs, _ = _fft_magnitude(signal, cutoff=0.25)
    assert np.all(freqs <= 0.25)


def test_grid_index_exact_match():
    assert _grid_index("lat", np.array([-40.0, 0.0, 40.0]), 0.0, label="p1") == 1


def test_grid_index_missing_raises():
    with pytest.raises(ValueError, match="not found in physics grid"):
        _grid_index("lat", np.array([-40.0, 0.0, 40.0]), 12.5, label="p1")


def test_harmonic_fft_writes_two_plots_and_two_npz(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("harmonic_fft")

    output = fn(_FakeModel(), id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()])

    assert len(output["plots"]) == 2
    assert len(output["data"]) == 2
    for p in output["plots"] + output["data"]:
        assert (tmp_path / p).is_file()
    assert any("altitude_scan" in p for p in output["plots"])
    assert any("latitude_scan" in p for p in output["plots"])


def test_harmonic_fft_empty_periods_raises(tmp_path):
    fn = get_kind_function("harmonic_fft")
    with pytest.raises(ValueError):
        fn(_FakeModel(), id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[])


def test_harmonic_fft_missing_lon_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("harmonic_fft")

    period = _one_period(lon_deg=45.0)  # not in _LON_VALUES
    with pytest.raises(ValueError, match="not found in physics grid"):
        fn(_FakeModel(), id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])


def test_harmonic_fft_missing_altitude_raises(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("harmonic_fft")

    period = _one_period(altitudes_km=[999.0])
    with pytest.raises(ValueError, match="not found in physics grid"):
        fn(_FakeModel(), id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])


def test_harmonic_fft_queries_rope_at_scan_values(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("harmonic_fft")
    model = _FakeModel()

    period = _one_period(lon_deg=0.0, lat_deg=0.0, altitudes_km=[150.0, 300.0],
                          alt_km=300.0, lats_deg=[-40.0, 40.0])
    fn(model, id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])

    queried_lats_alts = {(lat, alt) for _, _, lat, alt in model.query_calls}
    # altitude scan: lat fixed at 0.0, varying altitude
    assert (0.0, 150.0) in queried_lats_alts
    assert (0.0, 300.0) in queried_lats_alts
    # latitude scan: altitude fixed at 300.0, varying latitude
    assert (-40.0, 300.0) in queried_lats_alts
    assert (40.0, 300.0) in queried_lats_alts
    # every query used lon=0 -> lst == utc hour exactly
    for time, lst, _, _ in model.query_calls:
        hour = int(time.split(" ")[1].split(":")[0])
        assert lst == pytest.approx(float(hour))


def test_harmonic_fft_uses_suite_labels_as_panel_titles(tmp_path, monkeypatch):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("harmonic_fft")

    captured = []
    import rope_dev_tools.validation.checks.harmonic_fft as mod

    def fake_harmonic_fft_plot(panels, **kwargs):
        captured.append([p["title"] for p in panels])
        return kwargs.get("out_path")

    monkeypatch.setattr(mod, "harmonic_fft_plot", fake_harmonic_fft_plot)

    fn(_FakeModel(), id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()],
       physics_model_label="WAM", rope_model_label="ROPE-WAM-V1")

    assert captured == [["WAM", "ROPE-WAM-V1"], ["WAM", "ROPE-WAM-V1"]]


def test_harmonic_fft_npz_contains_expected_arrays(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("harmonic_fft")

    period = _one_period(altitudes_km=[150.0, 300.0], lats_deg=[-40.0, 0.0, 40.0])
    output = fn(_FakeModel(), id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[period])

    alt_npz_path = next(p for p in output["data"] if "altitude_scan" in p)
    with np.load(tmp_path / alt_npz_path) as npz:
        assert npz["physics_magnitude"].shape[0] == 2  # 2 altitudes
        assert npz["rope_magnitude"].shape == npz["physics_magnitude"].shape
        assert len(npz["altitudes_km"]) == 2

    lat_npz_path = next(p for p in output["data"] if "latitude_scan" in p)
    with np.load(tmp_path / lat_npz_path) as npz:
        assert npz["physics_magnitude"].shape[0] == 3  # 3 latitudes
        assert len(npz["lats_deg"]) == 3


def test_replot_harmonic_fft_reproduces_plots(tmp_path):
    _write_physics_npz(tmp_path / "phys.npz")
    fn = get_kind_function("harmonic_fft")
    output = fn(_FakeModel(), id="fft_test", out_dir=tmp_path, suite_dir=tmp_path, periods=[_one_period()])

    loaded = {}
    for path in output["data"]:
        with np.load(tmp_path / path) as npz:
            loaded[path] = {k: npz[k] for k in npz.files}

    replot_dir = tmp_path / "replotted"
    plots = replot_harmonic_fft(loaded, id="fft_test", out_dir=replot_dir)

    assert len(plots) == 2
    for p in plots:
        assert (replot_dir / p).is_file()
