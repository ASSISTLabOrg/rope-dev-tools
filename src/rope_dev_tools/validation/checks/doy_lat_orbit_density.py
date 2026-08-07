"""doy_lat_orbit_density — day-of-year vs latitude density, ascending/descending x satellite/physics/rope."""

from __future__ import annotations

import numpy as np

from rope_dev_tools.validation.checks import register_kind, register_replot
from rope_dev_tools.validation.data_artifacts import save_npz
from rope_dev_tools.validation.plots import doy_lat_plot
from rope_dev_tools.validation.statistics import compute_statistic_uncertainties, compute_statistics
from rope_dev_tools.validation.time_utils import resolve_path
from rope_dev_tools.validation.truth_data import load_ascending_track_csv, load_truth_csv


def _binned_mean(x, y, values, x_edges, y_edges) -> "np.ndarray":
    """Mean of values per (x, y) bin; NaN for empty bins."""
    total, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


@register_kind("doy_lat_orbit_density")
def doy_lat_orbit_density(
    model,
    *,
    id=None,
    start,
    end,
    satellite_track_csv,
    physics_model_track_csv,
    altitudes_km,
    lat_bin_deg,
    variable="density",
    statistics=None,
    unit=None,
    uncertainty=False,
    out_dir=None,
    suite_dir=None,
    **_,
) -> dict:
    """Per altitude/direction: bins satellite/physics/rope density by (day-of-year, lat) and plots each."""
    model.forecast(start, end, compute_uncertainty=uncertainty)

    sat = load_ascending_track_csv(resolve_path(suite_dir, satellite_track_csv))
    phys = load_truth_csv(resolve_path(suite_dir, physics_model_track_csv))
    if len(sat) != len(phys):
        raise ValueError(
            f"row count mismatch: {satellite_track_csv!r} has {len(sat)} rows, "
            f"{physics_model_track_csv!r} has {len(phys)}"
        )

    rope_results = [
        model.query(row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), row["lst"], row["lat"], row["alt_km"])
        for _, row in sat.iterrows()
    ]
    rope_values = np.asarray([r[variable] for r in rope_results])
    # only rope carries a per-point model uncertainty -- satellite/physics-model truth is exact.
    rope_uncert_values = np.asarray([r["uncertainty"] for r in rope_results]) if uncertainty else None

    doy = sat["datetime"].dt.dayofyear.to_numpy()
    lat = sat["lat"].to_numpy()
    alt = sat["alt_km"].to_numpy()
    ascending = sat["ascending"].astype(bool).to_numpy()
    doy_edges = np.arange(doy.min(), doy.max() + 2)
    lat_edges = np.arange(lat.min(), lat.max() + lat_bin_deg, lat_bin_deg)

    sources = {
        "satellite": sat[variable].to_numpy(),
        "physics_model": phys[variable].to_numpy(),
        "rope_model": rope_values,
    }

    plots, data_paths = [], []
    stats_by_altitude = {}

    for alt_km in altitudes_km:
        mask_alt = alt == alt_km
        grids = {}
        rope_uncert_grids = {}
        for direction, dir_mask in (("ascending", ascending), ("descending", ~ascending)):
            mask = mask_alt & dir_mask
            for label, values in sources.items():
                grid = _binned_mean(doy[mask], lat[mask], values[mask], doy_edges, lat_edges)
                grids[f"{direction}_{label}"] = grid
                plot_name = f"plots/{id}_{alt_km}km_{direction}_{label}.png"
                doy_lat_plot(grid, doy_edges=doy_edges, lat_edges=lat_edges,
                             out_path=f"{out_dir}/{plot_name}", title=f"{id} {alt_km}km {direction} {label}")
                plots.append(plot_name)
            if uncertainty:
                rope_uncert_grids[direction] = _binned_mean(
                    doy[mask], lat[mask], rope_uncert_values[mask], doy_edges, lat_edges,
                )

        alt_stats = {}
        for direction in ("ascending", "descending"):
            sat_grid = grids[f"{direction}_satellite"]
            valid = ~np.isnan(sat_grid)
            for comp_label, comp_key in (
                ("physics_vs_satellite", f"{direction}_physics_model"),
                ("rope_vs_satellite", f"{direction}_rope_model"),
            ):
                comp_grid = grids[comp_key]
                pair_valid = valid & ~np.isnan(comp_grid)
                stats = compute_statistics(comp_grid[pair_valid], sat_grid[pair_valid], statistics)
                if stats is not None:
                    alt_stats.setdefault(direction, {})[comp_label] = stats
                    if uncertainty and comp_label == "rope_vs_satellite":
                        uncert_grid = rope_uncert_grids[direction]
                        stat_uncerts = compute_statistic_uncertainties(
                            comp_grid[pair_valid], sat_grid[pair_valid], uncert_grid[pair_valid], statistics,
                        )
                        if stat_uncerts:
                            alt_stats[direction][f"{comp_label}_uncertainty"] = stat_uncerts
        if alt_stats:
            stats_by_altitude[f"{alt_km}km"] = alt_stats

        npz_name = f"{id}_{alt_km}km.npz"
        data_paths.append(save_npz(out_dir, npz_name, doy_edges=doy_edges, lat_edges=lat_edges, **grids))

    output = {"plots": plots, "data": data_paths}
    if stats_by_altitude:
        output["statistics"] = stats_by_altitude
    return output


@register_replot("doy_lat_orbit_density")
def replot_doy_lat_orbit_density(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: {array_name: np.ndarray}}, as produced by generate_validation_plots.py."""
    plots = []
    for path, npz in loaded.items():
        if not path.endswith(".npz"):
            continue
        alt_label = path.rsplit("/", 1)[-1].removeprefix(f"{id}_").removesuffix(".npz")
        doy_edges, lat_edges = npz["doy_edges"], npz["lat_edges"]
        for key, grid in npz.items():
            if key in ("doy_edges", "lat_edges"):
                continue
            plot_name = f"plots/{id}_{alt_label}_{key}.png"
            doy_lat_plot(grid, doy_edges=doy_edges, lat_edges=lat_edges,
                         out_path=f"{out_dir}/{plot_name}", title=f"{id} {alt_label} {key}")
            plots.append(plot_name)
    return plots
