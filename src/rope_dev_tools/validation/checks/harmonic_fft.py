"""harmonic_fft — FFT of density time series at fixed spatial points, physics vs rope, log-scale magnitude with diurnal-harmonic reference lines. No error metrics."""

from __future__ import annotations

import numpy as np

from rope_dev_tools.validation.checks import register_kind, register_replot
from rope_dev_tools.validation.data_artifacts import save_npz
from rope_dev_tools.validation.plots import harmonic_fft_plot
from rope_dev_tools.validation.time_utils import lst_values_for, parse_time, resolve_path

_HARMONIC_FREQS_PER_HOUR = (1.0 / 24, 1.0 / 12, 1.0 / 8, 1.0 / 6)
_LOWPASS_CUTOFF_PER_HOUR = 0.5


def _fft_magnitude(signal, *, cutoff: float) -> tuple:
    """De-biased rfft magnitude, low-pass filtered to (0, cutoff] cycles/hour."""
    signal = np.asarray(signal, dtype=float)
    debiased = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(debiased))
    freqs = np.fft.rfftfreq(len(signal), d=1.0)  # cycles/hour -- assumes hourly sampling
    mask = (freqs > 0) & (freqs <= cutoff)
    return freqs[mask], spectrum[mask]


def _grid_index(name: str, values: np.ndarray, query: float, *, label: str) -> int:
    """Index of the value closest-matching query; raises ValueError if none is close enough."""
    matches = np.nonzero(np.isclose(values, query))[0]
    if len(matches) == 0:
        raise ValueError(
            f"period {label!r}: {name} {query!r} not found in physics grid (available: "
            f"{sorted(set(np.round(values, 6).tolist()))})"
        )
    return int(matches[0])


@register_kind("harmonic_fft")
def harmonic_fft(
    model,
    *,
    id=None,
    periods,
    out_dir=None,
    suite_dir=None,
    physics_model_label=None,
    rope_model_label=None,
    **_,
) -> dict:
    """Per period: altitude scan and latitude scan FFT plots, physics vs rope, at fixed (lat, lon)/(alt, lon)."""
    if not periods:
        raise ValueError(f"check {id!r}: periods is empty")

    physics_model_label = physics_model_label or "physics"
    rope_model_label = rope_model_label or "rope"

    plots, data_paths = [], []
    for period in periods:
        label = period["label"]
        start, end = period["start"], period["end"]
        lon_deg = period["lon_deg"]
        lat_deg = period["lat_deg"]
        altitudes_km = period["altitudes_km"]
        alt_km = period["alt_km"]
        lats_deg = period["lats_deg"]
        physics_model_hourly_npz = period["physics_model_hourly_npz"]

        model.forecast(start, end)

        npz_path = resolve_path(suite_dir, physics_model_hourly_npz)
        with np.load(npz_path) as npz:
            phys_times = [str(t) for t in npz["times"]]
            phys_lon_values = np.asarray(npz["lon_values"], dtype=float)
            phys_n_lat = int(npz["n_lat"])
            phys_lat_min, phys_lat_max = float(npz["lat_min_deg"]), float(npz["lat_max_deg"])
            phys_altitudes = np.array([float(a) for a in npz["altitudes_km"]])
            phys_density = np.array(npz["density"])  # (H, A, n_lon, n_lat)
        phys_lat_values = np.linspace(phys_lat_min, phys_lat_max, phys_n_lat)

        start_dt, end_dt = parse_time(start), parse_time(end)
        time_mask = [start_dt <= parse_time(t) < end_dt for t in phys_times]
        if not any(time_mask):
            raise ValueError(
                f"check {id!r} period {label!r}: no timestamps in [{start}, {end}) found in "
                f"{physics_model_hourly_npz!r}"
            )
        phys_times = [t for t, m in zip(phys_times, time_mask) if m]
        phys_density = phys_density[np.array(time_mask)]

        lon_idx = _grid_index("lon_deg", phys_lon_values, lon_deg, label=label)

        # --- scenario 1: altitude scan at fixed (lat_deg, lon_deg) ---
        lat_idx = _grid_index("lat_deg", phys_lat_values, lat_deg, label=label)
        phys_series_alt, rope_series_alt, alt_freqs = {}, {}, None
        for alt in altitudes_km:
            alt_idx = _grid_index("altitude", phys_altitudes, alt, label=label)
            phys_signal = phys_density[:, alt_idx, lon_idx, lat_idx]
            rope_signal = [
                model.query(t, lst_values_for(lon_deg, t), lat_deg, alt)["density"] for t in phys_times
            ]
            f, m = _fft_magnitude(phys_signal, cutoff=_LOWPASS_CUTOFF_PER_HOUR)
            phys_series_alt[f"{alt}km"] = (f, m)
            f, m = _fft_magnitude(rope_signal, cutoff=_LOWPASS_CUTOFF_PER_HOUR)
            rope_series_alt[f"{alt}km"] = (f, m)
            alt_freqs = f

        alt_scan_plot = f"plots/{id}_{label}_altitude_scan.png"
        harmonic_fft_plot(
            [{"title": physics_model_label, "series": phys_series_alt},
             {"title": rope_model_label, "series": rope_series_alt}],
            harmonic_freqs_per_hour=list(_HARMONIC_FREQS_PER_HOUR),
            out_path=f"{out_dir}/{alt_scan_plot}",
            suptitle=f"{id} — {label} (lat={lat_deg}°, lon={lon_deg}°)",
        )
        plots.append(alt_scan_plot)

        data_paths.append(save_npz(
            out_dir, f"{id}_{label}_altitude_scan.npz",
            altitudes_km=np.array(altitudes_km, dtype=float), freqs=alt_freqs,
            physics_magnitude=np.stack([phys_series_alt[f"{a}km"][1] for a in altitudes_km]),
            rope_magnitude=np.stack([rope_series_alt[f"{a}km"][1] for a in altitudes_km]),
            lat_deg=lat_deg, lon_deg=lon_deg,
        ))

        # --- scenario 2: latitude scan at fixed (alt_km, lon_deg) ---
        fixed_alt_idx = _grid_index("alt_km", phys_altitudes, alt_km, label=label)
        phys_series_lat, rope_series_lat, lat_freqs = {}, {}, None
        for lat in lats_deg:
            scan_lat_idx = _grid_index("lats_deg", phys_lat_values, lat, label=label)
            phys_signal = phys_density[:, fixed_alt_idx, lon_idx, scan_lat_idx]
            rope_signal = [
                model.query(t, lst_values_for(lon_deg, t), lat, alt_km)["density"] for t in phys_times
            ]
            f, m = _fft_magnitude(phys_signal, cutoff=_LOWPASS_CUTOFF_PER_HOUR)
            phys_series_lat[f"lat={lat}°"] = (f, m)
            f, m = _fft_magnitude(rope_signal, cutoff=_LOWPASS_CUTOFF_PER_HOUR)
            rope_series_lat[f"lat={lat}°"] = (f, m)
            lat_freqs = f

        lat_scan_plot = f"plots/{id}_{label}_latitude_scan.png"
        harmonic_fft_plot(
            [{"title": physics_model_label, "series": phys_series_lat},
             {"title": rope_model_label, "series": rope_series_lat}],
            harmonic_freqs_per_hour=list(_HARMONIC_FREQS_PER_HOUR),
            out_path=f"{out_dir}/{lat_scan_plot}",
            suptitle=f"{id} — {label} (alt={alt_km}km, lon={lon_deg}°)",
        )
        plots.append(lat_scan_plot)

        data_paths.append(save_npz(
            out_dir, f"{id}_{label}_latitude_scan.npz",
            lats_deg=np.array(lats_deg, dtype=float), freqs=lat_freqs,
            physics_magnitude=np.stack([phys_series_lat[f"lat={lat}°"][1] for lat in lats_deg]),
            rope_magnitude=np.stack([rope_series_lat[f"lat={lat}°"][1] for lat in lats_deg]),
            alt_km=alt_km, lon_deg=lon_deg,
        ))

    return {"plots": plots, "data": data_paths}


@register_replot("harmonic_fft")
def replot_harmonic_fft(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: {array_name: np.ndarray}}, as produced by generate_validation_plots.py."""
    plots = []
    for path, npz in loaded.items():
        filename = path.rsplit("/", 1)[-1]
        prefix = f"{id}_"
        if not filename.startswith(prefix):
            continue

        freqs = npz["freqs"]
        phys_mag, rope_mag = npz["physics_magnitude"], npz["rope_magnitude"]

        if filename.endswith("_altitude_scan.npz"):
            label = filename[len(prefix):-len("_altitude_scan.npz")]
            names = [f"{v:g}km" for v in npz["altitudes_km"]]
            plot_name = f"plots/{id}_{label}_altitude_scan.png"
            suptitle = f"{id} — {label} (lat={float(npz['lat_deg'])}°, lon={float(npz['lon_deg'])}°)"
        elif filename.endswith("_latitude_scan.npz"):
            label = filename[len(prefix):-len("_latitude_scan.npz")]
            names = [f"lat={v:g}°" for v in npz["lats_deg"]]
            plot_name = f"plots/{id}_{label}_latitude_scan.png"
            suptitle = f"{id} — {label} (alt={float(npz['alt_km'])}km, lon={float(npz['lon_deg'])}°)"
        else:
            continue

        phys_series = {name: (freqs, phys_mag[i]) for i, name in enumerate(names)}
        rope_series = {name: (freqs, rope_mag[i]) for i, name in enumerate(names)}
        harmonic_fft_plot(
            [{"title": "physics", "series": phys_series}, {"title": "rope", "series": rope_series}],
            harmonic_freqs_per_hour=list(_HARMONIC_FREQS_PER_HOUR),
            out_path=f"{out_dir}/{plot_name}", suptitle=suptitle,
        )
        plots.append(plot_name)
    return plots
