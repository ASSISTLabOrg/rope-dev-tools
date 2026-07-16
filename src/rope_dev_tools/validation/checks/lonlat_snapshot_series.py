"""lonlat_snapshot_series — one forecast feeds both static NxM lon/lat snapshots and a full-horizon animation, physics + rope."""

from __future__ import annotations

import numpy as np

from rope_dev_tools.validation.checks import register_kind
from rope_dev_tools.validation.data_artifacts import save_npz
from rope_dev_tools.validation.plots import lonlat_animation, lonlat_plot
from rope_dev_tools.validation.statistics import compute_statistics
from rope_dev_tools.validation.time_utils import add_hours, resolve_path

_MAX_ANIMATION_HOURS = 72


def _resample_nearest(grid: "np.ndarray", dst_n_lst: int, dst_n_lat: int) -> "np.ndarray":
    """Nearest-index resample of an (n_lst, n_lat) grid onto a different (n_lst, n_lat) shape."""
    src_n_lst, src_n_lat = grid.shape
    lst_idx = (np.arange(dst_n_lst) / dst_n_lst * src_n_lst).astype(int) % src_n_lst
    lat_idx = np.clip(np.round(np.linspace(0, src_n_lat - 1, dst_n_lat)).astype(int), 0, src_n_lat - 1)
    return grid[np.ix_(lst_idx, lat_idx)]


@register_kind("lonlat_snapshot_series")
def lonlat_snapshot_series(
    model,
    *,
    id=None,
    start,
    horizon_hours,
    days,
    utc_hours,
    altitudes_km,
    include_snapshots=True,
    include_animation=True,
    physics_model_hourly_npz,
    animation_hours_step=1.0,
    statistics=None,
    unit=None,
    out_dir=None,
    suite_dir=None,
    **_,
) -> dict:
    if include_animation and horizon_hours > _MAX_ANIMATION_HOURS:
        raise ValueError(
            f"check {id!r}: include_animation requires horizon_hours <= {_MAX_ANIMATION_HOURS}, "
            f"got {horizon_hours}"
        )

    end = add_hours(start, horizon_hours)
    model.forecast(start, end)

    npz_path = resolve_path(suite_dir, physics_model_hourly_npz)
    with np.load(npz_path) as npz:
        phys_times = [str(t) for t in npz["times"]]
        phys_n_lst, phys_n_lat = int(npz["n_lst"]), int(npz["n_lat"])
        phys_lat_min, phys_lat_max = float(npz["lat_min_deg"]), float(npz["lat_max_deg"])
        phys_altitudes = [float(a) for a in npz["altitudes_km"]]
        phys_density = np.array(npz["density"])  # (H, A, n_lst, n_lat)

    rope_lat_range = (model.grid["lat_min_deg"], model.grid["lat_max_deg"])

    plots, data_paths = [], []
    stats_by_altitude = {}

    for alt_km in altitudes_km:
        if alt_km not in phys_altitudes:
            raise ValueError(f"altitude {alt_km} missing from {physics_model_hourly_npz!r}")
        alt_idx = phys_altitudes.index(alt_km)
        alt_stats = {}

        if include_snapshots:
            snap_times, rope_snaps, phys_snaps = [], [], []
            for day in days:
                day_times = [f"{day} {h:02d}:00:00" for h in utc_hours]
                phys_panels, rope_panels = [], []
                for t in day_times:
                    if t not in phys_times:
                        raise ValueError(f"time {t!r} missing from {physics_model_hourly_npz!r}")
                    phys_grid = phys_density[phys_times.index(t), alt_idx]
                    rope_grid = model.query_grid(t, alt_km)
                    phys_panels.append({"title": t, "grid": phys_grid})
                    rope_panels.append({"title": t, "grid": rope_grid})
                    snap_times.append(t)
                    phys_snaps.append(phys_grid)
                    rope_snaps.append(_resample_nearest(rope_grid, phys_n_lst, phys_n_lat))

                phys_plot = f"plots/{id}_{alt_km}km_{day}_physics.png"
                rope_plot = f"plots/{id}_{alt_km}km_{day}_rope.png"
                lonlat_plot(phys_panels, n_rows=1, n_cols=len(utc_hours), lat_range=(phys_lat_min, phys_lat_max),
                            out_path=f"{out_dir}/{phys_plot}", suptitle=f"{id} physics {alt_km}km {day}")
                lonlat_plot(rope_panels, n_rows=1, n_cols=len(utc_hours), lat_range=rope_lat_range,
                            out_path=f"{out_dir}/{rope_plot}", suptitle=f"{id} rope {alt_km}km {day}")
                plots += [phys_plot, rope_plot]

            stats = compute_statistics(np.array(rope_snaps), np.array(phys_snaps), statistics)
            if stats is not None:
                alt_stats["snapshot"] = {"model_vs_truth": stats}

            snap_npz_name = f"{id}_snapshots_{alt_km}km.npz"
            data_paths.append(save_npz(
                out_dir, snap_npz_name, times=np.array(snap_times),
                physics_density=np.array(phys_snaps), rope_density=np.array(rope_snaps),
                lat_min_deg=phys_lat_min, lat_max_deg=phys_lat_max,
            ))

        if include_animation:
            step = max(1, round(animation_hours_step))
            frame_idx = list(range(0, len(phys_times), step))
            anim_times = [phys_times[i] for i in frame_idx]
            phys_frames = [phys_density[i, alt_idx] for i in frame_idx]
            rope_frames = [model.query_grid(t, alt_km) for t in anim_times]

            anim_path = f"plots/{id}_{alt_km}km_animation.gif"
            lonlat_animation(
                [{"title": "physics", "frames": phys_frames}, {"title": "rope", "frames": rope_frames}],
                timestamps=anim_times, n_rows=1, n_cols=2, lat_range=(phys_lat_min, phys_lat_max),
                out_path=f"{out_dir}/{anim_path}", suptitle=f"{id} {alt_km}km",
            )
            plots.append(anim_path)

            resampled_rope = [_resample_nearest(g, phys_n_lst, phys_n_lat) for g in rope_frames]
            stats = compute_statistics(np.array(resampled_rope), np.array(phys_frames), statistics)
            if stats is not None:
                alt_stats["animation"] = {"model_vs_truth": stats}

            anim_npz_name = f"{id}_animation_{alt_km}km.npz"
            data_paths.append(save_npz(
                out_dir, anim_npz_name, times=np.array(anim_times),
                physics_density=np.array(phys_frames), rope_density=np.array(rope_frames),
                lat_min_deg=phys_lat_min, lat_max_deg=phys_lat_max,
            ))

        if alt_stats:
            stats_by_altitude[f"{alt_km}km"] = alt_stats

    output = {"plots": plots, "data": data_paths}
    if stats_by_altitude:
        output["statistics"] = stats_by_altitude
    return output


def replot_lonlat_snapshot_series(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: {array_name: np.ndarray}}, as produced by generate_validation_plots.py."""
    plots = []
    for path, npz in loaded.items():
        if "_snapshots_" not in path and "_animation_" not in path:
            continue
        filename = path.rsplit("/", 1)[-1]
        times = [str(t) for t in npz["times"]]
        lat_range = (float(npz["lat_min_deg"]), float(npz["lat_max_deg"]))
        phys_frames, rope_frames = list(npz["physics_density"]), list(npz["rope_density"])

        if "_snapshots_" in filename:
            alt_label = filename.split("_snapshots_")[-1].removesuffix(".npz")
            phys_panels = [{"title": t, "grid": g} for t, g in zip(times, phys_frames)]
            rope_panels = [{"title": t, "grid": g} for t, g in zip(times, rope_frames)]
            phys_plot, rope_plot = f"plots/{id}_{alt_label}_physics.png", f"plots/{id}_{alt_label}_rope.png"
            lonlat_plot(phys_panels, n_rows=1, n_cols=len(times), lat_range=lat_range,
                        out_path=f"{out_dir}/{phys_plot}", suptitle=f"{id} physics {alt_label}")
            lonlat_plot(rope_panels, n_rows=1, n_cols=len(times), lat_range=lat_range,
                        out_path=f"{out_dir}/{rope_plot}", suptitle=f"{id} rope {alt_label}")
            plots += [phys_plot, rope_plot]
        else:
            alt_label = filename.split("_animation_")[-1].removesuffix(".npz")
            anim_plot = f"plots/{id}_{alt_label}_animation.gif"
            lonlat_animation(
                [{"title": "physics", "frames": phys_frames}, {"title": "rope", "frames": rope_frames}],
                timestamps=times, n_rows=1, n_cols=2, lat_range=lat_range,
                out_path=f"{out_dir}/{anim_plot}", suptitle=f"{id} {alt_label}",
            )
            plots.append(anim_plot)
    return plots
