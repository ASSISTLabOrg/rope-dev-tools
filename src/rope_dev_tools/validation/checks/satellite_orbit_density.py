"""satellite_orbit_density — satellite/physics/rope density along a satellite track, one line plot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rope_dev_tools.validation.checks import passes_threshold, register_kind
from rope_dev_tools.validation.data_artifacts import save_csv
from rope_dev_tools.validation.plots import line_plot
from rope_dev_tools.validation.statistics import compute_statistics, format_statistics_text
from rope_dev_tools.validation.time_utils import resolve_path
from rope_dev_tools.validation.truth_data import load_truth_csv


@register_kind("satellite_orbit_density")
def satellite_orbit_density(
    model,
    *,
    id=None,
    start,
    end,
    satellite_track_csv,
    physics_model_track_csv,
    variable="density",
    statistics=None,
    threshold=None,
    unit=None,
    out_dir=None,
    suite_dir=None,
    **_,
) -> dict:
    model.forecast(start, end)

    sat = load_truth_csv(resolve_path(suite_dir, satellite_track_csv))
    phys = load_truth_csv(resolve_path(suite_dir, physics_model_track_csv))
    if len(sat) != len(phys):
        raise ValueError(
            f"row count mismatch: {satellite_track_csv!r} has {len(sat)} rows, "
            f"{physics_model_track_csv!r} has {len(phys)}"
        )

    rope_values = [
        model.query(row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), row["lst"], row["lat"], row["alt_km"])[variable]
        for _, row in sat.iterrows()
    ]

    comparison = pd.DataFrame({
        "datetime": sat["datetime"], "lst": sat["lst"], "lat": sat["lat"], "alt_km": sat["alt_km"],
        "satellite_density": sat[variable], "physics_density": phys[variable], "rope_density": rope_values,
    })
    if "ascending" in sat.columns:
        comparison["ascending"] = sat["ascending"]
    data_path = save_csv(out_dir, f"{id}.csv", comparison)

    rope_arr = np.asarray(rope_values)
    sat_arr = comparison["satellite_density"].to_numpy()
    phys_arr = comparison["physics_density"].to_numpy()

    stats = {}
    rope_vs_sat = compute_statistics(rope_arr, sat_arr, statistics)
    if rope_vs_sat is not None:
        stats["rope_vs_satellite"] = rope_vs_sat
    phys_vs_sat = compute_statistics(phys_arr, sat_arr, statistics)
    if phys_vs_sat is not None:
        stats["physics_vs_satellite"] = phys_vs_sat

    panel = {
        "title": id or "satellite_orbit_density",
        "ylabel": unit or variable,
        "series": {
            "satellite": (comparison["datetime"], sat_arr),
            "physics_model": (comparison["datetime"], phys_arr),
            "rope_model": (comparison["datetime"], rope_arr),
        },
        "stats_text": format_statistics_text(rope_vs_sat),
    }
    plot_name = f"plots/{id}.png"
    line_plot([panel], out_path=f"{out_dir}/{plot_name}", suptitle=id)

    value = float(np.sqrt(np.mean(np.square(rope_arr - sat_arr))))
    passed = passes_threshold(value, threshold) if threshold else None

    output = {"value": value, "unit": unit, "passed": passed, "plots": [plot_name], "data": [data_path]}
    if stats:
        output["statistics"] = stats
    return output


def replot_satellite_orbit_density(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: DataFrame}, as produced by generate_validation_plots.py."""
    data = loaded[f"validation_data/{id}.csv"]
    panel = {
        "title": id,
        "ylabel": unit or "density",
        "series": {
            "satellite": (data["datetime"], data["satellite_density"]),
            "physics_model": (data["datetime"], data["physics_density"]),
            "rope_model": (data["datetime"], data["rope_density"]),
        },
    }
    plot_name = f"plots/{id}.png"
    line_plot([panel], out_path=f"{out_dir}/{plot_name}", suptitle=id)
    return [plot_name]
