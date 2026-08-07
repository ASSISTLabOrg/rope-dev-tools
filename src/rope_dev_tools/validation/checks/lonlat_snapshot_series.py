"""lonlat_snapshot_series — per period (per start_delta), one forecast feeds static NxM lon/lat snapshots and a full-horizon animation, physics + rope."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from rope_dev_tools.validation.checks import (
    delta_label,
    delta_stat_key,
    delta_suffix,
    register_kind,
    register_replot,
)
from rope_dev_tools.validation.data_artifacts import save_npz
from rope_dev_tools.validation.plots import lonlat_animation, lonlat_plot
from rope_dev_tools.validation.statistics import compute_statistic_uncertainties, compute_statistics
from rope_dev_tools.validation.time_utils import (
    add_hours,
    lst_values_for,
    parse_time,
    resolve_path,
    resolve_start_delta,
)

_MAX_ANIMATION_HOURS = 72
_MAX_SNAPSHOT_PANELS_PER_PLOT = 4


def _calendar_days(start_dt, end_dt) -> list:
    """'YYYY-MM-DD' for each date from start_dt to end_dt, inclusive."""
    days = []
    d = start_dt.date()
    while d <= end_dt.date():
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


def _write_snapshot_plots(panels: list, *, lat_range, x_range, out_dir, base_path: str,
                           suptitle: str, vmin, vmax) -> list:
    """Splits panels into <= _MAX_SNAPSHOT_PANELS_PER_PLOT-panel plots, suffixed _partN if more than one."""
    chunks = [panels[i:i + _MAX_SNAPSHOT_PANELS_PER_PLOT]
              for i in range(0, len(panels), _MAX_SNAPSHOT_PANELS_PER_PLOT)]
    plots = []
    for i, chunk in enumerate(chunks):
        suffix = "" if len(chunks) == 1 else f"_part{i + 1}"
        plot_name = f"{base_path}{suffix}.png"
        lonlat_plot(chunk, n_rows=1, n_cols=len(chunk), lat_range=lat_range, x_range=x_range,
                    xlabel="Longitude (deg)", vmin=vmin, vmax=vmax,
                    out_path=f"{out_dir}/{plot_name}", suptitle=suptitle)
        plots.append(plot_name)
    return plots


def _query_grid_maybe_uncert(model, t, alt_km, lst_values, lat_values, uncertainty: bool) -> tuple:
    """(density, uncertainty_or_None) at the given axis values."""
    result = model.query_grid_at(t, alt_km, lst_values, lat_values, include_uncertainty=uncertainty)
    return (result["density"], result["uncertainty"]) if uncertainty else (result, None)


def _per_frame_statistics(rope_frames: list, phys_frames: list, names: list) -> dict:
    """Each statistic in names, computed independently per matching frame pair."""
    per_frame = [compute_statistics(np.asarray(r), np.asarray(p), names) for r, p in zip(rope_frames, phys_frames)]
    return {name: np.array([pf[name] for pf in per_frame]) for name in names}


def _per_frame_statistic_uncertainties(rope_frames: list, rope_uncert_frames: list, phys_frames: list,
                                        names: list) -> "dict | None":
    """Each statistic's uncertainty in names that has one registered, computed independently per frame; None if none do."""
    per_frame = [
        compute_statistic_uncertainties(np.asarray(r), np.asarray(p), np.asarray(u), names)
        for r, u, p in zip(rope_frames, rope_uncert_frames, phys_frames)
    ]
    available = [name for name in names if all(name in pf for pf in per_frame)]
    if not available:
        return None
    return {name: np.array([pf[name] for pf in per_frame]) for name in available}


def _shared_color_range(*grids: list) -> tuple:
    """(min, max) across every grid in every list passed in; (None, None) if all empty."""
    all_values = [g for grids_list in grids for g in grids_list]
    if not all_values:
        return None, None
    stacked = np.concatenate([np.asarray(g).ravel() for g in all_values])
    return float(np.min(stacked)), float(np.max(stacked))


@register_kind("lonlat_snapshot_series")
def lonlat_snapshot_series(
    model,
    *,
    id=None,
    periods,
    altitudes_km,
    statistics=None,
    unit=None,
    uncertainty=False,
    out_dir=None,
    suite_dir=None,
    physics_model_label=None,
    rope_model_label=None,
    **_,
) -> dict:
    """Per period/altitude/start_delta: snapshot plots and/or a full-horizon animation, physics + rope."""
    if not periods:
        raise ValueError(f"check {id!r}: periods is empty")

    physics_model_label = physics_model_label or "physics"
    rope_model_label = rope_model_label or "rope"

    plots, data_paths = [], []
    stats_by_period = {}

    for period in periods:
        label = period["label"]
        start = period["start"]
        horizon_hours = period["horizon_hours"]
        utc_hours = period["utc_hours"]
        include_snapshots = period.get("include_snapshots", True)
        include_animation = period.get("include_animation", True)
        animation_hours_step = period.get("animation_hours_step", 1.0)
        physics_model_hourly_npz = period["physics_model_hourly_npz"]
        start_deltas = period.get("start_deltas", [0])
        n_deltas = len(start_deltas)
        widest_delta = min(start_deltas)
        plot_stats = period.get("plot_stats", False)
        plot_stat_uncertainty = period.get("plot_stat_uncertainty", False)
        if plot_stats and not statistics:
            raise ValueError(
                f"check {id!r} period {label!r}: plot_stats=true requires a non-empty "
                f"'statistics' list on the check"
            )
        if plot_stat_uncertainty and not plot_stats:
            raise ValueError(f"check {id!r} period {label!r}: plot_stat_uncertainty=true requires plot_stats=true")
        if plot_stat_uncertainty and not uncertainty:
            raise ValueError(f"check {id!r} period {label!r}: plot_stat_uncertainty=true requires uncertainty=true")

        end = add_hours(start, horizon_hours)
        start_dt, end_dt = parse_time(start), parse_time(end)

        npz_path = resolve_path(suite_dir, physics_model_hourly_npz)
        with np.load(npz_path) as npz:
            phys_times = [str(t) for t in npz["times"]]
            phys_lon_values = np.asarray(npz["lon_values"], dtype=float)
            phys_n_lat = int(npz["n_lat"])
            phys_lat_min, phys_lat_max = float(npz["lat_min_deg"]), float(npz["lat_max_deg"])
            phys_altitudes = [float(a) for a in npz["altitudes_km"]]
            phys_density = np.array(npz["density"])  # (H, A, n_lon, n_lat)

        time_mask = [start_dt <= parse_time(t) <= end_dt for t in phys_times]
        if not any(time_mask):
            raise ValueError(
                f"check {id!r} period {label!r}: no timestamps in [{start}, {end}] found in "
                f"{physics_model_hourly_npz!r}"
            )
        phys_times = [t for t, m in zip(phys_times, time_mask) if m]
        phys_density = phys_density[np.array(time_mask)]
        phys_lat_values = np.linspace(phys_lat_min, phys_lat_max, phys_n_lat)
        rope_lat_min, rope_lat_max = model.grid["lat_min_deg"], model.grid["lat_max_deg"]
        lat_mask = (phys_lat_values >= rope_lat_min) & (phys_lat_values <= rope_lat_max)
        if not lat_mask.any():
            raise ValueError(
                f"check {id!r} period {label!r}: physics lat range [{phys_lat_min}, {phys_lat_max}] "
                f"does not overlap ROPE's grid range [{rope_lat_min}, {rope_lat_max}]"
            )
        phys_lat_values = phys_lat_values[lat_mask]
        phys_density = phys_density[..., lat_mask]
        phys_lat_min, phys_lat_max = float(phys_lat_values.min()), float(phys_lat_values.max())
        phys_lon_min, phys_lon_max = float(phys_lon_values.min()), float(phys_lon_values.max())

        for alt_km in altitudes_km:
            if alt_km not in phys_altitudes:
                raise ValueError(
                    f"check {id!r} period {label!r}: altitude {alt_km} missing from "
                    f"{physics_model_hourly_npz!r}"
                )

        days = _calendar_days(start_dt, end_dt)
        all_day_times = [f"{day} {h:02d}:00:00" for day in days for h in utc_hours]

        gathered = {}  # (alt_km, delta) -> {"snap_times", "rope_snaps", "rope_snaps_uncert", "anim_times", "rope_frames", "rope_frames_uncert"}
        for delta in start_deltas:
            forecast_start, query_start_dt = resolve_start_delta(start, end, delta)
            animated_hours = (end_dt - query_start_dt).total_seconds() / 3600.0
            if include_animation and animated_hours > _MAX_ANIMATION_HOURS:
                raise ValueError(
                    f"check {id!r} period {label!r} start_delta {delta!r}h: include_animation "
                    f"requires the animated window (query_start to end) <= "
                    f"{_MAX_ANIMATION_HOURS}h, got {animated_hours}h"
                )
            model.forecast(forecast_start, end, compute_uncertainty=uncertainty)

            for alt_km in altitudes_km:
                alt_idx = phys_altitudes.index(alt_km)

                snap_times, rope_snaps, rope_snaps_uncert = [], [], []
                if include_snapshots:
                    day_times = [t for t in all_day_times if query_start_dt <= parse_time(t) <= end_dt]
                    if not day_times:
                        raise ValueError(
                            f"check {id!r} period {label!r} start_delta {delta!r}h: none of "
                            f"utc_hours {utc_hours!r} fall within [{query_start_dt}, {end}] "
                            f"across days {days!r}"
                        )
                    for t in day_times:
                        if t not in phys_times:
                            raise ValueError(
                                f"check {id!r} period {label!r}: time {t!r} missing from "
                                f"{physics_model_hourly_npz!r}"
                            )
                        density, uncert = _query_grid_maybe_uncert(
                            model, t, alt_km, lst_values_for(phys_lon_values, t), phys_lat_values, uncertainty,
                        )
                        snap_times.append(t)
                        rope_snaps.append(density)
                        if uncertainty:
                            rope_snaps_uncert.append(uncert)

                anim_times, rope_frames, rope_frames_uncert = [], [], []
                if include_animation:
                    step = max(1, round(animation_hours_step))
                    frame_idx = [i for i in range(0, len(phys_times), step)
                                 if parse_time(phys_times[i]) >= query_start_dt]
                    anim_times = [phys_times[i] for i in frame_idx]
                    for t in anim_times:
                        density, uncert = _query_grid_maybe_uncert(
                            model, t, alt_km, lst_values_for(phys_lon_values, t), phys_lat_values, uncertainty,
                        )
                        rope_frames.append(density)
                        if uncertainty:
                            rope_frames_uncert.append(uncert)

                gathered[(alt_km, delta)] = {
                    "snap_times": snap_times, "rope_snaps": rope_snaps, "rope_snaps_uncert": rope_snaps_uncert,
                    "anim_times": anim_times, "rope_frames": rope_frames, "rope_frames_uncert": rope_frames_uncert,
                }

        for alt_km in altitudes_km:
            alt_idx = phys_altitudes.index(alt_km)
            alt_stats = {}

            widest = gathered[(alt_km, widest_delta)]
            phys_snaps_display = [phys_density[phys_times.index(t), alt_idx] for t in widest["snap_times"]]
            phys_frames_display = [phys_density[phys_times.index(t), alt_idx] for t in widest["anim_times"]]
            all_rope_snaps = [g for delta in start_deltas for g in gathered[(alt_km, delta)]["rope_snaps"]]
            all_rope_frames = [g for delta in start_deltas for g in gathered[(alt_km, delta)]["rope_frames"]]
            vmin, vmax = _shared_color_range(phys_snaps_display, all_rope_snaps, phys_frames_display, all_rope_frames)

            lat_range = (phys_lat_min, phys_lat_max)
            lon_range = (phys_lon_min, phys_lon_max)

            if include_snapshots:
                phys_panels = [{"title": t, "grid": g} for t, g in zip(widest["snap_times"], phys_snaps_display)]
                plots += _write_snapshot_plots(
                    phys_panels, lat_range=lat_range, x_range=lon_range, out_dir=out_dir,
                    base_path=f"plots/{id}_{alt_km}km_{label}_physics",
                    suptitle=f"{id} {physics_model_label} {alt_km}km {label}", vmin=vmin, vmax=vmax,
                )

                snapshot_stats = {}
                for delta in start_deltas:
                    g = gathered[(alt_km, delta)]
                    rope_panels = [{"title": t, "grid": grid} for t, grid in zip(g["snap_times"], g["rope_snaps"])]
                    plots += _write_snapshot_plots(
                        rope_panels, lat_range=lat_range, x_range=lon_range, out_dir=out_dir,
                        base_path=f"plots/{id}_{alt_km}km_{label}_rope{delta_suffix(delta, n_deltas=n_deltas)}",
                        suptitle=delta_label(f"{id} {rope_model_label} {alt_km}km {label}", delta, n_deltas=n_deltas),
                        vmin=vmin, vmax=vmax,
                    )
                    aligned_phys = [phys_density[phys_times.index(t), alt_idx] for t in g["snap_times"]]
                    stats = compute_statistics(np.array(g["rope_snaps"]), np.array(aligned_phys), statistics)
                    stat_uncerts = None
                    if uncertainty and stats is not None:
                        stat_uncerts = compute_statistic_uncertainties(
                            np.array(g["rope_snaps"]), np.array(aligned_phys), np.array(g["rope_snaps_uncert"]),
                            statistics,
                        )
                    if stats is not None:
                        entry = {"model_vs_truth": stats}
                        if stat_uncerts:
                            entry["model_vs_truth_uncertainty"] = stat_uncerts
                        snapshot_stats[delta_stat_key(delta)] = entry

                    snap_npz_name = (
                        f"{id}_snapshots_{label}_{alt_km}km{delta_suffix(delta, n_deltas=n_deltas)}.npz"
                    )
                    snap_save_kwargs = dict(
                        times=np.array(g["snap_times"]),
                        physics_density=np.array(aligned_phys), rope_density=np.array(g["rope_snaps"]),
                        lat_min_deg=phys_lat_min, lat_max_deg=phys_lat_max,
                        lon_min_deg=phys_lon_min, lon_max_deg=phys_lon_max,
                    )
                    if uncertainty:
                        snap_save_kwargs["rope_uncert"] = np.array(g["rope_snaps_uncert"])
                    data_paths.append(save_npz(out_dir, snap_npz_name, **snap_save_kwargs))
                if snapshot_stats:
                    alt_stats["snapshot"] = snapshot_stats

            if include_animation:
                animation_stats = {}
                for delta in start_deltas:
                    g = gathered[(alt_km, delta)]
                    aligned_phys_frames = [phys_density[phys_times.index(t), alt_idx] for t in g["anim_times"]]
                    anim_path = f"plots/{id}_{alt_km}km_{label}_animation{delta_suffix(delta, n_deltas=n_deltas)}.gif"
                    stats_series = (
                        _per_frame_statistics(g["rope_frames"], aligned_phys_frames, statistics)
                        if plot_stats else None
                    )
                    stats_uncertainty_series = (
                        _per_frame_statistic_uncertainties(
                            g["rope_frames"], g["rope_frames_uncert"], aligned_phys_frames, statistics,
                        ) if plot_stat_uncertainty else None
                    )
                    lonlat_animation(
                        [{"title": physics_model_label, "frames": aligned_phys_frames},
                         {"title": delta_label(rope_model_label, delta, n_deltas=n_deltas), "frames": g["rope_frames"]}],
                        timestamps=g["anim_times"], n_rows=1, n_cols=2, lat_range=lat_range,
                        x_range=lon_range, xlabel="Longitude (deg)", vmin=vmin, vmax=vmax,
                        out_path=f"{out_dir}/{anim_path}",
                        suptitle=delta_label(f"{id} {alt_km}km {label}", delta, n_deltas=n_deltas),
                        stats_series=stats_series,
                        stats_uncertainty_series=stats_uncertainty_series,
                    )
                    plots.append(anim_path)

                    stats = compute_statistics(np.array(g["rope_frames"]), np.array(aligned_phys_frames), statistics)
                    stat_uncerts = None
                    if uncertainty and stats is not None:
                        stat_uncerts = compute_statistic_uncertainties(
                            np.array(g["rope_frames"]), np.array(aligned_phys_frames),
                            np.array(g["rope_frames_uncert"]), statistics,
                        )
                    if stats is not None:
                        entry = {"model_vs_truth": stats}
                        if stat_uncerts:
                            entry["model_vs_truth_uncertainty"] = stat_uncerts
                        animation_stats[delta_stat_key(delta)] = entry

                    anim_npz_name = (
                        f"{id}_animation_{label}_{alt_km}km{delta_suffix(delta, n_deltas=n_deltas)}.npz"
                    )
                    anim_save_kwargs = dict(
                        times=np.array(g["anim_times"]),
                        physics_density=np.array(aligned_phys_frames), rope_density=np.array(g["rope_frames"]),
                        lat_min_deg=phys_lat_min, lat_max_deg=phys_lat_max,
                        lon_min_deg=phys_lon_min, lon_max_deg=phys_lon_max,
                    )
                    if uncertainty:
                        anim_save_kwargs["rope_uncert"] = np.array(g["rope_frames_uncert"])
                    data_paths.append(save_npz(out_dir, anim_npz_name, **anim_save_kwargs))
                if animation_stats:
                    alt_stats["animation"] = animation_stats

            if alt_stats:
                stats_by_period.setdefault(label, {})[f"{alt_km}km"] = alt_stats

    output = {"plots": plots, "data": data_paths}
    if stats_by_period:
        output["statistics"] = stats_by_period
    return output


@register_replot("lonlat_snapshot_series")
def replot_lonlat_snapshot_series(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: {array_name: np.ndarray}}, as produced by generate_validation_plots.py."""
    plots = []
    for path, npz in loaded.items():
        if "_snapshots_" not in path and "_animation_" not in path:
            continue
        filename = path.rsplit("/", 1)[-1]
        times = [str(t) for t in npz["times"]]
        lat_range = (float(npz["lat_min_deg"]), float(npz["lat_max_deg"]))
        lon_range = (float(npz["lon_min_deg"]), float(npz["lon_max_deg"]))
        phys_frames, rope_frames = list(npz["physics_density"]), list(npz["rope_density"])
        vmin, vmax = _shared_color_range(phys_frames, rope_frames)

        if "_snapshots_" in filename:
            alt_label = filename.split("_snapshots_")[-1].removesuffix(".npz")
            phys_panels = [{"title": t, "grid": g} for t, g in zip(times, phys_frames)]
            rope_panels = [{"title": t, "grid": g} for t, g in zip(times, rope_frames)]
            phys_plots = _write_snapshot_plots(
                phys_panels, lat_range=lat_range, x_range=lon_range, out_dir=out_dir,
                base_path=f"plots/{id}_{alt_label}_physics", suptitle=f"{id} physics {alt_label}",
                vmin=vmin, vmax=vmax,
            )
            rope_plots = _write_snapshot_plots(
                rope_panels, lat_range=lat_range, x_range=lon_range, out_dir=out_dir,
                base_path=f"plots/{id}_{alt_label}_rope", suptitle=f"{id} rope {alt_label}",
                vmin=vmin, vmax=vmax,
            )
            plots += phys_plots + rope_plots
        else:
            alt_label = filename.split("_animation_")[-1].removesuffix(".npz")
            anim_plot = f"plots/{id}_{alt_label}_animation.gif"
            lonlat_animation(
                [{"title": "physics", "frames": phys_frames}, {"title": "rope", "frames": rope_frames}],
                timestamps=times, n_rows=1, n_cols=2, lat_range=lat_range,
                x_range=lon_range, xlabel="Longitude (deg)", vmin=vmin, vmax=vmax,
                out_path=f"{out_dir}/{anim_plot}", suptitle=f"{id} {alt_label}",
            )
            plots.append(anim_plot)
    return plots
