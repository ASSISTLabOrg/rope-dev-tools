"""avg_density_vs_time — grid-average density vs time at fixed altitudes, truth vs model, any period length."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rope_dev_tools.validation.checks import register_kind
from rope_dev_tools.validation.data_artifacts import save_csv
from rope_dev_tools.validation.plots import line_plot
from rope_dev_tools.validation.statistics import compute_statistics, format_statistics_text
from rope_dev_tools.validation.time_utils import parse_time, resolve_path
from rope_dev_tools.validation.truth_data import load_avg_density_csv


@register_kind("avg_density_vs_time")
def avg_density_vs_time(
    model,
    *,
    id=None,
    periods,
    altitudes_km,
    statistics=None,
    unit=None,
    requires_exported_model=False,
    out_dir=None,
    suite_dir=None,
    **_,
) -> dict:
    if requires_exported_model and model.backend_name != "exported_dir":
        raise ValueError(
            f"check {id!r} sets requires_exported_model=true but is running against a "
            f"{model.backend_name!r} model interface; re-run against a real exported model directory"
        )

    rows = []
    for period in periods:
        model.forecast(period["start"], period["end"])

        physics_avg_csv = period["physics_avg_csv"]
        paths = physics_avg_csv if isinstance(physics_avg_csv, list) else [physics_avg_csv]
        truth = load_avg_density_csv([resolve_path(suite_dir, p) for p in paths])

        start_dt, end_dt = parse_time(period["start"]), parse_time(period["end"])
        truth = truth[(truth["datetime"] >= start_dt) & (truth["datetime"] < end_dt)]
        if truth.empty:
            raise ValueError(
                f"no truth rows in [{period['start']}, {period['end']}) loaded from {paths!r} "
                f"(period {period['label']!r})"
            )

        for alt_km in altitudes_km:
            subset = truth[truth["alt_km"] == alt_km]
            if subset.empty:
                raise ValueError(
                    f"altitude {alt_km} missing from {paths!r} within "
                    f"[{period['start']}, {period['end']}) (period {period['label']!r})"
                )
            for _, row in subset.sort_values("datetime").iterrows():
                grid = model.query_grid(row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), alt_km)
                rows.append({
                    "period": period["label"], "datetime": row["datetime"], "alt_km": alt_km,
                    "truth_density": row["density"], "model_density": float(np.mean(grid)),
                })

    comparison = pd.DataFrame(rows)
    data_path = save_csv(out_dir, f"{id}.csv", comparison)

    plots = []
    stats_by_period = {}
    for period in periods:
        period_rows = comparison[comparison["period"] == period["label"]]
        panels = []
        for alt_km in altitudes_km:
            alt_rows = period_rows[period_rows["alt_km"] == alt_km].sort_values("datetime")
            stats = compute_statistics(
                alt_rows["model_density"].to_numpy(), alt_rows["truth_density"].to_numpy(), statistics,
            )
            if stats is not None:
                stats_by_period.setdefault(period["label"], {})[f"{alt_km}km"] = {"model_vs_truth": stats}
            panels.append({
                "title": f"{alt_km} km",
                "ylabel": unit or "density",
                "series": {
                    "truth": (alt_rows["datetime"], alt_rows["truth_density"]),
                    "model": (alt_rows["datetime"], alt_rows["model_density"]),
                },
                "stats_text": format_statistics_text(stats),
            })
        plot_name = f"plots/{id}_{period['label']}.png"
        line_plot(panels, out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {period['label']}")
        plots.append(plot_name)

    output = {"plots": plots, "data": [data_path]}
    if stats_by_period:
        output["statistics"] = stats_by_period
    return output


def replot_avg_density_vs_time(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: DataFrame}, as produced by generate_validation_plots.py."""
    data = loaded[f"validation_data/{id}.csv"]
    periods = list(dict.fromkeys(data["period"]))
    altitudes_km = sorted(data["alt_km"].unique())

    plots = []
    for period in periods:
        period_rows = data[data["period"] == period]
        panels = []
        for alt_km in altitudes_km:
            alt_rows = period_rows[period_rows["alt_km"] == alt_km].sort_values("datetime")
            panels.append({
                "title": f"{alt_km} km",
                "ylabel": unit or "density",
                "series": {
                    "truth": (alt_rows["datetime"], alt_rows["truth_density"]),
                    "model": (alt_rows["datetime"], alt_rows["model_density"]),
                },
            })
        plot_name = f"plots/{id}_{period}.png"
        line_plot(panels, out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {period}")
        plots.append(plot_name)
    return plots
