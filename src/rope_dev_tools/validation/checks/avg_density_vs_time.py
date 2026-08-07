"""avg_density_vs_time — grid-average density vs time at fixed altitudes, truth vs model, any period length."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rope_dev_tools.validation.checks import delta_label, delta_stat_key, register_kind, register_replot
from rope_dev_tools.validation.data_artifacts import save_csv
from rope_dev_tools.validation.plots import line_plot
from rope_dev_tools.validation.statistics import (
    compute_statistic_uncertainties,
    compute_statistics,
    format_statistics_text,
    uncertainty_of_mean,
)
from rope_dev_tools.validation.time_utils import parse_time, resolve_path, resolve_start_delta
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
    uncertainty=False,
    plot_uncertainty=False,
    out_dir=None,
    suite_dir=None,
    physics_model_label=None,
    rope_model_label=None,
    **_,
) -> dict:
    """Per period/altitude/start_delta: forecasts, queries the grid mean, and plots vs truth."""
    physics_model_label = physics_model_label or "truth"
    rope_model_label = rope_model_label or "model"
    if requires_exported_model and model.backend_name != "exported_dir":
        raise ValueError(
            f"check {id!r} sets requires_exported_model=true but is running against a "
            f"{model.backend_name!r} model interface; re-run against a real exported model directory"
        )
    if plot_uncertainty and not uncertainty:
        raise ValueError(f"check {id!r} sets plot_uncertainty=true but uncertainty=false; nothing to plot")

    rows = []
    for period in periods:
        start_deltas = period.get("start_deltas", [0])

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
            if truth[truth["alt_km"] == alt_km].empty:
                raise ValueError(
                    f"altitude {alt_km} missing from {paths!r} within "
                    f"[{period['start']}, {period['end']}) (period {period['label']!r})"
                )

        for delta in start_deltas:
            forecast_start, query_start_dt = resolve_start_delta(period["start"], period["end"], delta)
            model.forecast(forecast_start, period["end"], compute_uncertainty=uncertainty)

            for alt_km in altitudes_km:
                subset = truth[(truth["alt_km"] == alt_km) & (truth["datetime"] >= query_start_dt)]
                if subset.empty:
                    raise ValueError(
                        f"period {period['label']!r} altitude {alt_km}: start_delta {delta!r}h "
                        f"leaves no truth rows in [{query_start_dt}, {period['end']})"
                    )
                for _, row in subset.sort_values("datetime").iterrows():
                    result = model.query_grid(row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), alt_km,
                                               include_uncertainty=uncertainty)
                    density_grid = result["density"] if uncertainty else result
                    model_uncert = uncertainty_of_mean(result["uncertainty"]) if uncertainty else None
                    rows.append({
                        "period": period["label"], "datetime": row["datetime"], "alt_km": alt_km,
                        "start_delta": delta, "truth_density": row["density"],
                        "model_density": float(np.mean(density_grid)), "model_uncert": model_uncert,
                    })

    comparison = pd.DataFrame(rows)
    data_path = save_csv(out_dir, f"{id}.csv", comparison)

    plots = []
    stats_by_period = {}
    for period in periods:
        start_deltas = period.get("start_deltas", [0])
        n_deltas = len(start_deltas)
        widest_delta = min(start_deltas)

        period_rows = comparison[comparison["period"] == period["label"]]
        panels = []
        for alt_km in altitudes_km:
            alt_rows = period_rows[period_rows["alt_km"] == alt_km]
            truth_rows = alt_rows[alt_rows["start_delta"] == widest_delta].sort_values("datetime")
            series = {physics_model_label: (truth_rows["datetime"], truth_rows["truth_density"])}

            stats_lines = []
            for delta in start_deltas:
                delta_rows = alt_rows[alt_rows["start_delta"] == delta].sort_values("datetime")
                model_density = delta_rows["model_density"]
                label = delta_label(rope_model_label, delta, n_deltas=n_deltas)
                series[label] = (
                    (delta_rows["datetime"], model_density, delta_rows["model_uncert"]) if plot_uncertainty
                    else (delta_rows["datetime"], model_density)
                )
                stats = compute_statistics(model_density.to_numpy(), delta_rows["truth_density"].to_numpy(), statistics)
                stat_uncerts = None
                if uncertainty and stats is not None:
                    stat_uncerts = compute_statistic_uncertainties(
                        model_density.to_numpy(), delta_rows["truth_density"].to_numpy(),
                        delta_rows["model_uncert"].to_numpy(), statistics,
                    )
                if stats is not None:
                    entry = {"model_vs_truth": stats}
                    if stat_uncerts:
                        entry["model_vs_truth_uncertainty"] = stat_uncerts
                    stats_by_period.setdefault(period["label"], {}).setdefault(f"{alt_km}km", {})[
                        delta_stat_key(delta)
                    ] = entry
                    text = format_statistics_text(stats, stat_uncerts)
                    if n_deltas > 1:
                        text = "\n".join(f"Δ{delta:+d}h {line}" for line in text.split("\n"))
                    stats_lines.append(text)

            panels.append({
                "title": f"{alt_km} km",
                "ylabel": unit or "density",
                "series": series,
                "stats_text": "\n".join(stats_lines) if stats_lines else None,
            })
        plot_name = f"plots/{id}_{period['label']}.png"
        line_plot(panels, out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {period['label']}")
        plots.append(plot_name)

    output = {"plots": plots, "data": [data_path]}
    if stats_by_period:
        output["statistics"] = stats_by_period
    return output


@register_replot("avg_density_vs_time")
def replot_avg_density_vs_time(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: DataFrame}, as produced by generate_validation_plots.py. Shows the saved
    uncertainty band automatically if the saved data has a (non-null) model_uncert column."""
    data = loaded[f"validation_data/{id}.csv"]
    periods = list(dict.fromkeys(data["period"]))
    altitudes_km = sorted(data["alt_km"].unique())
    has_uncert = "model_uncert" in data.columns and data["model_uncert"].notna().any()

    plots = []
    for period in periods:
        period_rows = data[data["period"] == period]
        start_deltas = sorted(period_rows["start_delta"].unique())
        n_deltas = len(start_deltas)
        widest_delta = min(start_deltas)

        panels = []
        for alt_km in altitudes_km:
            alt_rows = period_rows[period_rows["alt_km"] == alt_km]
            truth_rows = alt_rows[alt_rows["start_delta"] == widest_delta].sort_values("datetime")
            series = {"truth": (truth_rows["datetime"], truth_rows["truth_density"])}
            for delta in start_deltas:
                delta_rows = alt_rows[alt_rows["start_delta"] == delta].sort_values("datetime")
                label = delta_label("model", delta, n_deltas=n_deltas)
                series[label] = (
                    (delta_rows["datetime"], delta_rows["model_density"], delta_rows["model_uncert"]) if has_uncert
                    else (delta_rows["datetime"], delta_rows["model_density"])
                )
            panels.append({
                "title": f"{alt_km} km",
                "ylabel": unit or "density",
                "series": series,
            })
        plot_name = f"plots/{id}_{period}.png"
        line_plot(panels, out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {period}")
        plots.append(plot_name)
    return plots
