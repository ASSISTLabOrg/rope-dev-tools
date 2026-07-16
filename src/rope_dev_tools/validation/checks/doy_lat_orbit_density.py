"""doy_lat_orbit_density — day-of-year vs latitude density, ascending/descending x satellite/physics/rope."""

from __future__ import annotations

import numpy as np

from rope_dev_tools.validation.checks import register_kind
from rope_dev_tools.validation.data_artifacts import save_npz
from rope_dev_tools.validation.plots import doy_lat_plot
from rope_dev_tools.validation.statistics import compute_statistics
from rope_dev_tools.validation.time_utils import resolve_path
from rope_dev_tools.validation.truth_data import load_ascending_track_csv, load_truth_csv


def _binned_mean(x, y, values, x_edges, y_edges) -> "np.ndarray":
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
    out_dir=None,
    suite_dir=None,
    **_,
) -> dict:
    model.forecast(start, end)

    sat = load_ascending_track_csv(resolve_path(suite_dir, satellite_track_csv))
    phys = load_truth_csv(resolve_path(suite_dir, physics_model_track_csv))
    if len(sat) != len(phys):
        raise ValueError(
            f"row count mismatch: {satellite_track_csv!r} has {len(sat)} rows, "
            f"{physics_model_track_csv!r} has {len(phys)}"
        )

    rope_values = np.asarray([
        model.query(row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), row["lst"], row["lat"], row["alt_km"])[variable]
        for _, row in sat.iterrows()
    ])

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
        for direction, dir_mask in (("ascending", ascending), ("descending", ~ascending)):
            mask = mask_alt & dir_mask
            for label, values in sources.items():
                grid = _binned_mean(doy[mask], lat[mask], values[mask], doy_edges, lat_edges)
                grids[f"{direction}_{label}"] = grid
                plot_name = f"plots/{id}_{alt_km}km_{direction}_{label}.png"
                doy_lat_plot(grid, doy_edges=doy_edges, lat_edges=lat_edges,
                             out_path=f"{out_dir}/{plot_name}", title=f"{id} {alt_km}km {direction} {label}")
                plots.append(plot_name)

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
        if alt_stats:
            stats_by_altitude[f"{alt_km}km"] = alt_stats

        npz_name = f"{id}_{alt_km}km.npz"
        data_paths.append(save_npz(out_dir, npz_name, doy_edges=doy_edges, lat_edges=lat_edges, **grids))

    output = {"plots": plots, "data": data_paths}
    if stats_by_altitude:
        output["statistics"] = stats_by_altitude
    return output


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
