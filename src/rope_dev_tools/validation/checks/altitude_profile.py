"""altitude_profile — global-average density vs altitude at selected time slices, physics vs rope."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from rope_dev_tools.validation.checks import delta_label, delta_stat_key, delta_suffix, register_kind, register_replot
from rope_dev_tools.validation.data_artifacts import save_npz
from rope_dev_tools.validation.statistics import compute_statistics, format_statistics_text
from rope_dev_tools.validation.time_utils import add_hours, parse_time, resolve_path, resolve_start_delta


@register_kind("altitude_profile")
def altitude_profile(
    model,
    *,
    id=None,
    periods,
    altitude_cutouts=None,
    statistics=None,
    unit=None,
    out_dir=None,
    suite_dir=None,
    physics_model_label=None,
    rope_model_label=None,
    **_,
) -> dict:
    """Per period/utc_hour: global-average density vs altitude, physics and rope on the same panel."""
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
        physics_model_hourly_npz = period["physics_model_hourly_npz"]
        start_deltas = period.get("start_deltas", [0])
        n_deltas = len(start_deltas)
        widest_delta = min(start_deltas)

        end = add_hours(start, horizon_hours)
        start_dt, end_dt = parse_time(start), parse_time(end)

        npz_path = resolve_path(suite_dir, physics_model_hourly_npz)
        with np.load(npz_path) as npz:
            phys_times = [str(t) for t in npz["times"]]
            phys_lon_values = np.asarray(npz["lon_values"], dtype=float)
            phys_n_lat = int(npz["n_lat"])
            phys_lat_min, phys_lat_max = float(npz["lat_min_deg"]), float(npz["lat_max_deg"])
            phys_altitudes = np.asarray(npz["altitudes_km"], dtype=float)
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

        rope_n_lst = model.grid["n_lst"]
        rope_n_lat = model.grid["n_lat"]
        rope_lst_values = np.linspace(0.0, 24.0, rope_n_lst, endpoint=False)
        rope_lat_values = np.linspace(rope_lat_min, rope_lat_max, rope_n_lat)

        days = []
        d = start_dt.date()
        while d <= end_dt.date():
            days.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        all_day_times = [f"{day} {h:02d}:00:00" for day in days for h in utc_hours]

        for delta in start_deltas:
            forecast_start, query_start_dt = resolve_start_delta(start, end, delta)
            model.forecast(forecast_start, end)

            day_times = [t for t in all_day_times if query_start_dt <= parse_time(t) < end_dt]
            if not day_times:
                raise ValueError(
                    f"check {id!r} period {label!r} start_delta {delta!r}h: none of "
                    f"utc_hours {utc_hours!r} fall within [{query_start_dt}, {end}]"
                )

            panels = []
            phys_profiles_all, rope_profiles_all = [], []
            for t in day_times:
                if t not in phys_times:
                    raise ValueError(
                        f"check {id!r} period {label!r}: time {t!r} missing from "
                        f"{physics_model_hourly_npz!r}"
                    )
                ti = phys_times.index(t)

                # Physics: global mean over (lon, lat) at each altitude.
                phys_profile = np.array([
                    np.mean(phys_density[ti, ai]) for ai in range(len(phys_altitudes))
                ])

                # ROPE: query at each altitude on its native grid, no interpolation.
                rope_profile = np.array([
                    np.mean(model.query_grid_at(t, float(alt_km), rope_lst_values, rope_lat_values))
                    for alt_km in phys_altitudes
                ])

                phys_profiles_all.append(phys_profile)
                rope_profiles_all.append(rope_profile)

                stats = compute_statistics(rope_profile, phys_profile, statistics)
                stats_text = format_statistics_text(stats)

                rope_label = delta_label(rope_model_label, delta, n_deltas=n_deltas)
                panels.append({
                    "title": t,
                    "xlabel": unit or "density",
                    "ylabel": "altitude (km)",
                    "series": {
                        physics_model_label: (phys_profile, phys_altitudes),
                        rope_label: (rope_profile, phys_altitudes),
                    },
                    "residual": (100.0 * np.abs(phys_profile - rope_profile) / np.where(phys_profile != 0, phys_profile, np.nan), phys_altitudes),
                    "stats_text": stats_text,
                })

            if statistics:
                period_stats = {}
                for i, t in enumerate(day_times):
                    snap_stats = compute_statistics(rope_profiles_all[i], phys_profiles_all[i], statistics)
                    if snap_stats is not None:
                        period_stats[t] = {"model_vs_truth": snap_stats}
                if period_stats:
                    stats_by_period.setdefault(label, {})[delta_stat_key(delta)] = period_stats

            npz_name = f"{id}_{label}{delta_suffix(delta, n_deltas=n_deltas)}.npz"
            data_paths.append(save_npz(
                out_dir, npz_name,
                times=np.array(day_times),
                altitudes_km=phys_altitudes,
                physics_profiles=np.array(phys_profiles_all),
                rope_profiles=np.array(rope_profiles_all),
            ))

            plot_name = f"plots/{id}_{label}{delta_suffix(delta, n_deltas=n_deltas)}.png"
            _altitude_profile_plot(
                panels, out_path=f"{out_dir}/{plot_name}",
                suptitle=delta_label(f"{id} — {label}", delta, n_deltas=n_deltas),
            )
            plots.append(plot_name)

            for cutout in (altitude_cutouts or []):
                alt_min, alt_max = cutout["min_km"], cutout["max_km"]
                cut_mask = (phys_altitudes >= alt_min) & (phys_altitudes <= alt_max)
                cut_alts = phys_altitudes[cut_mask]
                cut_panels = []
                for i, t in enumerate(day_times):
                    cut_phys = np.array(phys_profiles_all[i])[cut_mask]
                    cut_rope = np.array(rope_profiles_all[i])[cut_mask]
                    cut_stats = compute_statistics(cut_rope, cut_phys, statistics)
                    cut_stats_text = format_statistics_text(cut_stats)
                    rope_lbl = delta_label(rope_model_label, delta, n_deltas=n_deltas)
                    cut_panels.append({
                        "title": t,
                        "xlabel": unit or "density",
                        "ylabel": "altitude (km)",
                        "series": {
                            physics_model_label: (cut_phys, cut_alts),
                            rope_lbl: (cut_rope, cut_alts),
                        },
                        "residual": (100.0 * np.abs(cut_phys - cut_rope) / np.where(cut_phys != 0, cut_phys, np.nan), cut_alts),
                        "stats_text": cut_stats_text,
                    })
                cut_label = cutout["label"]
                cut_plot = f"plots/{id}_{label}{delta_suffix(delta, n_deltas=n_deltas)}_{cut_label}.png"
                cut_suptitle = delta_label(f"{id} — {label} ({alt_min:.0f}–{alt_max:.0f} km)", delta, n_deltas=n_deltas)
                plot_alt_min = cutout.get("plot_alt_min_km", alt_min)
                _altitude_profile_plot(
                    cut_panels, out_path=f"{out_dir}/{cut_plot}",
                    suptitle=cut_suptitle, alt_range=(plot_alt_min, alt_max),
                    log_x=cutout.get("log_x", True),
                )
                plots.append(cut_plot)

    output = {"plots": plots, "data": data_paths}
    if stats_by_period:
        output["statistics"] = stats_by_period
    return output


def _least_crowded_corner(axes):
    """Picks the axes corner with the fewest drawn points across all axes sharing the panel."""
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    inv = axes[0].transAxes.inverted()
    pts = []
    for ax in axes:
        for line in ax.get_lines():
            xd, yd = line.get_xdata(), line.get_ydata()
            if len(xd) == 0:
                continue
            display_pts = ax.transData.transform(np.column_stack([xd, yd]))
            normed = inv.transform(display_pts)
            pts.append(normed)
    if not pts:
        return 0.05, 0.05, "left", "bottom"
    all_pts = np.vstack(pts)

    corners = [
        (0.05, 0.05, "left", "bottom"),
        (0.95, 0.05, "right", "bottom"),
        (0.05, 0.95, "left", "top"),
        (0.95, 0.95, "right", "top"),
    ]
    counts = []
    for cx, cy, _, _ in corners:
        in_quad = ((all_pts[:, 0] < 0.35) if cx < 0.5 else (all_pts[:, 0] > 0.65)) & \
                  ((all_pts[:, 1] < 0.35) if cy < 0.5 else (all_pts[:, 1] > 0.65))
        counts.append(int(np.sum(in_quad)))
    best = int(np.argmin(counts))
    return corners[best]


def _altitude_profile_plot(panels, *, out_path, suptitle=None, alt_range=None, log_x=True):
    """Horizontal subplots, one per time slice; density on x-axis, altitude on y-axis."""
    from rope_dev_tools.validation.plots._common import savefig, use_agg_backend

    plt = use_agg_backend()

    # Pre-scan for consistent |bias| axis across all panels.
    bias_max = 0.0
    for panel in panels:
        residual = panel.get("residual")
        if residual is not None:
            bm = float(np.nanmax(residual[0]))
            if np.isfinite(bm):
                bias_max = max(bias_max, bm)
    bias_lim = bias_max * 1.05 if bias_max > 0 else None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 5.0), squeeze=False, sharey=True)
    deferred_text = []
    for ax, panel in zip(axes[0, :], panels):
        for lbl, (x, y) in panel["series"].items():
            ax.plot(x, y, label=lbl, linewidth=1.8)
        if log_x:
            ax.set_xscale("log")
        if alt_range is not None:
            ax.set_ylim(alt_range[0], alt_range[1])
        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel(panel.get("xlabel", "density"), fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        panel_axes = [ax]
        residual = panel.get("residual")
        if residual is not None:
            rx, ry = residual
            ax2 = ax.twiny()
            ax2.plot(rx, ry, color="tab:red", linewidth=1.2, linestyle="--", label="|bias| %", alpha=0.8)
            if bias_lim is not None:
                ax2.set_xlim(0, bias_lim)
            ax2.set_xlabel("|bias| %", fontsize=10, color="tab:red")
            ax2.tick_params(axis="x", labelsize=9, colors="tab:red")
            ax2.spines["top"].set_color("tab:red")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)
            panel_axes.append(ax2)
        else:
            ax.legend(loc="best", fontsize=9)
        stats_text = panel.get("stats_text")
        if stats_text:
            deferred_text.append((ax, panel_axes, stats_text))
    axes[0, 0].set_ylabel(panels[0].get("ylabel", "altitude (km)"), fontsize=12)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.canvas.draw()
    for ax, panel_axes, stats_text in deferred_text:
        sx, sy, sha, sva = _least_crowded_corner(panel_axes)
        ax.text(sx, sy, stats_text, transform=ax.transAxes, ha=sha, va=sva,
                fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})
    return savefig(fig, out_path)


@register_replot("altitude_profile")
def replot_altitude_profile(
    loaded: dict, *, id, out_dir, unit=None, altitude_cutouts=None,
    physics_model_label=None, rope_model_label=None,
    statistics=("bias", "rmse", "std"), **_,
) -> list:
    """Regenerates altitude profile plots from saved npz data."""
    phys_lbl = physics_model_label or "physics"
    rope_lbl = rope_model_label or "rope"
    plots = []
    for path, npz in loaded.items():
        filename = path.rsplit("/", 1)[-1]
        if not filename.startswith(f"{id}_") or not filename.endswith(".npz"):
            continue

        label = filename[len(f"{id}_"):-len(".npz")]
        times = [str(t) for t in npz["times"]]
        altitudes_km = npz["altitudes_km"]
        phys_profiles = npz["physics_profiles"]
        rope_profiles = npz["rope_profiles"]

        panels = []
        for i, t in enumerate(times):
            stats = compute_statistics(rope_profiles[i], phys_profiles[i], list(statistics))
            stats_text = format_statistics_text(stats)
            panels.append({
                "title": t,
                "xlabel": unit or "density",
                "ylabel": "altitude (km)",
                "series": {
                    phys_lbl: (phys_profiles[i], altitudes_km),
                    rope_lbl: (rope_profiles[i], altitudes_km),
                },
                "residual": (100.0 * np.abs(phys_profiles[i] - rope_profiles[i]) / np.where(phys_profiles[i] != 0, phys_profiles[i], np.nan), altitudes_km),
                "stats_text": stats_text,
            })

        plot_name = f"plots/{id}_{label}.png"
        _altitude_profile_plot(panels, out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {label}")
        plots.append(plot_name)

        for cutout in (altitude_cutouts or []):
            alt_min, alt_max = cutout["min_km"], cutout["max_km"]
            cut_mask = (altitudes_km >= alt_min) & (altitudes_km <= alt_max)
            cut_alts = altitudes_km[cut_mask]
            cut_panels = []
            for i, t in enumerate(times):
                cut_phys = phys_profiles[i][cut_mask]
                cut_rope = rope_profiles[i][cut_mask]
                cut_stats = compute_statistics(cut_rope, cut_phys, list(statistics))
                cut_stats_text = format_statistics_text(cut_stats)
                cut_panels.append({
                    "title": t,
                    "xlabel": unit or "density",
                    "ylabel": "altitude (km)",
                    "series": {
                        phys_lbl: (cut_phys, cut_alts),
                        rope_lbl: (cut_rope, cut_alts),
                    },
                    "residual": (100.0 * np.abs(cut_phys - cut_rope) / np.where(cut_phys != 0, cut_phys, np.nan), cut_alts),
                    "stats_text": cut_stats_text,
                })
            cut_label = cutout["label"]
            cut_plot = f"plots/{id}_{label}_{cut_label}.png"
            cut_suptitle = f"{id} — {label} ({alt_min:.0f}–{alt_max:.0f} km)"
            plot_alt_min = cutout.get("plot_alt_min_km", alt_min)
            _altitude_profile_plot(
                cut_panels, out_path=f"{out_dir}/{cut_plot}",
                suptitle=cut_suptitle, alt_range=(plot_alt_min, alt_max),
                log_x=cutout.get("log_x", True),
            )
            plots.append(cut_plot)
    return plots
