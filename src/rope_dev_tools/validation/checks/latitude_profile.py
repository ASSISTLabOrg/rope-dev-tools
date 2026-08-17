"""latitude_profile — zonal-average density vs latitude at selected altitudes, physics vs rope."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from rope_dev_tools.validation.checks import delta_label, delta_stat_key, delta_suffix, register_kind, register_replot
from rope_dev_tools.validation.checks.altitude_profile import _least_crowded_corner
from rope_dev_tools.validation.data_artifacts import save_npz
from rope_dev_tools.validation.statistics import compute_statistics, format_statistics_text
from rope_dev_tools.validation.time_utils import add_hours, parse_time, resolve_path, resolve_start_delta


@register_kind("latitude_profile")
def latitude_profile(
    model,
    *,
    id=None,
    periods,
    altitudes_km,
    statistics=None,
    unit=None,
    out_dir=None,
    suite_dir=None,
    physics_model_label=None,
    rope_model_label=None,
    **_,
) -> dict:
    """Per period/altitude/utc_hour: zonal-average density vs latitude, physics and rope on the same panel."""
    if not periods:
        raise ValueError(f"check {id!r}: periods is empty")
    if not altitudes_km:
        raise ValueError(f"check {id!r}: altitudes_km is empty")

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

        end = add_hours(start, horizon_hours)
        start_dt, end_dt = parse_time(start), parse_time(end)

        npz_path = resolve_path(suite_dir, physics_model_hourly_npz)
        with np.load(npz_path) as npz:
            phys_times = [str(t) for t in npz["times"]]
            phys_n_lat = int(npz["n_lat"])
            phys_lat_min, phys_lat_max = float(npz["lat_min_deg"]), float(npz["lat_max_deg"])
            phys_altitudes = np.asarray(npz["altitudes_km"], dtype=float)
            phys_density = np.array(npz["density"])

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
        rope_lst_values = np.linspace(0.0, 24.0, rope_n_lst, endpoint=False)

        for alt_km in altitudes_km:
            if alt_km not in phys_altitudes:
                raise ValueError(
                    f"check {id!r} period {label!r}: altitude {alt_km} km missing from "
                    f"{physics_model_hourly_npz!r}"
                )

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

            for alt_km in altitudes_km:
                alt_idx = list(phys_altitudes).index(alt_km)
                panels = []
                phys_profiles_all, rope_profiles_all = [], []

                for t in day_times:
                    if t not in phys_times:
                        raise ValueError(
                            f"check {id!r} period {label!r}: time {t!r} missing from "
                            f"{physics_model_hourly_npz!r}"
                        )
                    ti = phys_times.index(t)

                    # Physics: zonal mean over longitude at this altitude.
                    phys_profile = np.mean(phys_density[ti, alt_idx], axis=0)

                    # ROPE: query at native LST grid and physics lat grid, zonal mean over LST.
                    rope_grid = model.query_grid_at(t, float(alt_km), rope_lst_values, phys_lat_values)
                    rope_profile = np.mean(rope_grid, axis=0)

                    phys_profiles_all.append(phys_profile)
                    rope_profiles_all.append(rope_profile)

                    stats = compute_statistics(rope_profile, phys_profile, statistics)
                    stats_text = format_statistics_text(stats)
                    rope_lbl = delta_label(rope_model_label, delta, n_deltas=n_deltas)
                    residual = 100.0 * np.abs(phys_profile - rope_profile) / np.where(
                        phys_profile != 0, phys_profile, np.nan
                    )
                    panels.append({
                        "title": t,
                        "ylabel": unit or "density",
                        "series": {
                            physics_model_label: (phys_lat_values, phys_profile),
                            rope_lbl: (phys_lat_values, rope_profile),
                        },
                        "residual": (phys_lat_values, residual),
                        "stats_text": stats_text,
                    })

                if statistics:
                    period_stats = {}
                    for i, t_str in enumerate(day_times):
                        snap_stats = compute_statistics(
                            rope_profiles_all[i], phys_profiles_all[i], statistics,
                        )
                        if snap_stats is not None:
                            period_stats[t_str] = {"model_vs_truth": snap_stats}
                    if period_stats:
                        stats_by_period.setdefault(label, {}).setdefault(
                            f"{alt_km}km", {},
                        )[delta_stat_key(delta)] = period_stats

                npz_name = f"{id}_{label}_{alt_km}km{delta_suffix(delta, n_deltas=n_deltas)}.npz"
                data_paths.append(save_npz(
                    out_dir, npz_name,
                    times=np.array(day_times),
                    lat_values=phys_lat_values,
                    altitude_km=np.float64(alt_km),
                    physics_profiles=np.array(phys_profiles_all),
                    rope_profiles=np.array(rope_profiles_all),
                ))

                plot_name = f"plots/{id}_{label}_{alt_km}km{delta_suffix(delta, n_deltas=n_deltas)}.png"
                _latitude_profile_plot(
                    panels, out_path=f"{out_dir}/{plot_name}",
                    suptitle=delta_label(
                        f"{id} — {label} ({alt_km:.0f} km)", delta, n_deltas=n_deltas,
                    ),
                )
                plots.append(plot_name)

    output = {"plots": plots, "data": data_paths}
    if stats_by_period:
        output["statistics"] = stats_by_period
    return output


def _latitude_profile_plot(panels, *, out_path, suptitle=None, lat_range=None):
    """Horizontal subplots; latitude on x-axis, density on left y-axis (log), |bias| % on right y-axis."""
    from rope_dev_tools.validation.plots._common import savefig, use_agg_backend

    plt = use_agg_backend()

    # Pre-scan for consistent |bias| axis across all panels.
    bias_max = 0.0
    for panel in panels:
        residual = panel.get("residual")
        if residual is not None:
            bm = float(np.nanmax(residual[1]))
            if np.isfinite(bm):
                bias_max = max(bias_max, bm)
    bias_lim = bias_max * 1.05 if bias_max > 0 else None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 5.0), squeeze=False, sharey=True)
    deferred_text = []
    for i, (ax, panel) in enumerate(zip(axes[0, :], panels)):
        for lbl, (x, y) in panel["series"].items():
            ax.plot(x, y, label=lbl, linewidth=1.8)
        ax.set_yscale("log")
        if lat_range is not None:
            ax.set_xlim(lat_range[0], lat_range[1])
        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel("latitude (deg)", fontsize=12)
        if i == 0:
            ax.set_ylabel(panel.get("ylabel", "density"), fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        panel_axes = [ax]
        residual = panel.get("residual")
        if residual is not None:
            rlat, rbias = residual
            ax2 = ax.twinx()
            ax2.plot(rlat, rbias, color="tab:red", linewidth=1.2, linestyle="--",
                     label="|bias| %", alpha=0.8)
            if bias_lim is not None:
                ax2.set_ylim(0, bias_lim)
            ax2.set_ylabel("|bias| %", fontsize=10, color="tab:red")
            ax2.tick_params(axis="y", labelsize=9, colors="tab:red")
            ax2.spines["right"].set_color("tab:red")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)
            panel_axes.append(ax2)
        else:
            ax.legend(loc="best", fontsize=9)
        stats_text = panel.get("stats_text")
        if stats_text:
            deferred_text.append((ax, panel_axes, stats_text))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.canvas.draw()
    for ax, panel_axes, stats_text in deferred_text:
        sx, sy, sha, sva = _least_crowded_corner(panel_axes)
        ax.text(sx, sy, stats_text, transform=ax.transAxes, ha=sha, va=sva,
                fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})
    return savefig(fig, out_path)


@register_replot("latitude_profile")
def replot_latitude_profile(
    loaded: dict, *, id, out_dir, unit=None,
    physics_model_label=None, rope_model_label=None,
    statistics=("bias", "rmse", "std"), **_,
) -> list:
    """Regenerates latitude profile plots from saved npz data."""
    phys_lbl = physics_model_label or "physics"
    rope_lbl = rope_model_label or "rope"
    plots = []
    for path, npz in loaded.items():
        filename = path.rsplit("/", 1)[-1]
        if not filename.startswith(f"{id}_") or not filename.endswith(".npz"):
            continue

        label = filename[len(f"{id}_"):-len(".npz")]
        times = [str(t) for t in npz["times"]]
        lat_values = npz["lat_values"]
        alt_km = float(npz["altitude_km"])
        phys_profiles = npz["physics_profiles"]
        rope_profiles = npz["rope_profiles"]

        panels = []
        for i, t in enumerate(times):
            stats = compute_statistics(rope_profiles[i], phys_profiles[i], list(statistics))
            stats_text = format_statistics_text(stats)
            residual = 100.0 * np.abs(phys_profiles[i] - rope_profiles[i]) / np.where(
                phys_profiles[i] != 0, phys_profiles[i], np.nan
            )
            panels.append({
                "title": t,
                "ylabel": unit or "density",
                "series": {
                    phys_lbl: (lat_values, phys_profiles[i]),
                    rope_lbl: (lat_values, rope_profiles[i]),
                },
                "residual": (lat_values, residual),
                "stats_text": stats_text,
            })

        plot_name = f"plots/{id}_{label}.png"
        _latitude_profile_plot(
            panels, out_path=f"{out_dir}/{plot_name}",
            suptitle=f"{id} — {label} ({alt_km:.0f} km)",
        )
        plots.append(plot_name)
    return plots
