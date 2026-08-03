"""satellite_orbit_density — satellite/physics/rope density along a satellite track, one line plot per period, any number of periods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rope_dev_tools.validation.checks import delta_label, delta_stat_key, passes_threshold, register_kind
from rope_dev_tools.validation.data_artifacts import save_csv
from rope_dev_tools.validation.plots import line_plot
from rope_dev_tools.validation.statistics import compute_statistics, format_statistics_text
from rope_dev_tools.validation.time_utils import parse_time, resolve_path, resolve_start_delta
from rope_dev_tools.validation.truth_data import load_truth_csv

_DENSITY_COLUMNS = ("satellite_density", "physics_density", "rope_density")


def _orbit_average(period_rows: "pd.DataFrame", label: str) -> "pd.DataFrame":
    df = period_rows.reset_index(drop=True)

    if "ascending" in df.columns:
        ascending = df["ascending"].astype(bool).to_numpy()
    elif "lat" in df.columns:
        if len(df) < 2:
            raise ValueError(
                f"period {label!r}: orbit_averaged=true needs at least 2 rows to derive orbit "
                f"direction from latitude"
            )
        lat = df["lat"].to_numpy(dtype=float)
        d_lat = np.diff(lat)
        ascending = np.concatenate([d_lat >= 0, [d_lat[-1] >= 0]])
    else:
        raise ValueError(
            f"period {label!r}: orbit_averaged=true requires an 'ascending' column, or a 'lat' "
            f"column to derive direction from, in the satellite track CSV"
        )

    starts_new_orbit = np.zeros(len(ascending), dtype=bool)
    starts_new_orbit[0] = ascending[0]
    starts_new_orbit[1:] = ascending[1:] & ~ascending[:-1]
    orbit_id = np.cumsum(starts_new_orbit)
    df = df[orbit_id > 0].copy()
    df["_ascending"] = ascending[orbit_id > 0]
    df["_orbit_id"] = orbit_id[orbit_id > 0]

    if df.empty:
        raise ValueError(f"period {label!r}: orbit_averaged=true found no complete orbits")

    grouped = df.groupby("_orbit_id")
    complete = grouped.filter(lambda g: g["_ascending"].any() and (~g["_ascending"]).any())
    if complete.empty:
        raise ValueError(f"period {label!r}: orbit_averaged=true found no complete orbits")

    agg = {"datetime": "mean", **{col: "mean" for col in _DENSITY_COLUMNS}}
    return complete.groupby("_orbit_id").agg(agg).reset_index(drop=True)


@register_kind("satellite_orbit_density")
def satellite_orbit_density(
    model,
    *,
    id=None,
    periods,
    variable="density",
    statistics=None,
    threshold=None,
    unit=None,
    out_dir=None,
    suite_dir=None,
    physics_model_label=None,
    rope_model_label=None,
    satellite_label=None,
    **_,
) -> dict:
    if not periods:
        raise ValueError(f"check {id!r}: periods is empty")

    physics_model_label = physics_model_label or "physics_model"
    rope_model_label = rope_model_label or "rope_model"
    satellite_label = satellite_label or "satellite"

    rows = []
    for period in periods:
        start_deltas = period.get("start_deltas", [0])

        sat = load_truth_csv(resolve_path(suite_dir, period["satellite_track_csv"]))
        phys = load_truth_csv(resolve_path(suite_dir, period["physics_model_track_csv"]))

        start_dt, end_dt = parse_time(period["start"]), parse_time(period["end"])
        sat = sat[(sat["datetime"] >= start_dt) & (sat["datetime"] < end_dt)].reset_index(drop=True)
        phys = phys[(phys["datetime"] >= start_dt) & (phys["datetime"] < end_dt)].reset_index(drop=True)

        if len(sat) != len(phys):
            raise ValueError(
                f"period {period['label']!r}: row count mismatch after filtering to "
                f"[{period['start']}, {period['end']}): {period['satellite_track_csv']!r} has "
                f"{len(sat)} rows, {period['physics_model_track_csv']!r} has {len(phys)}"
            )
        if sat.empty:
            raise ValueError(
                f"period {period['label']!r}: no rows in [{period['start']}, {period['end']}) "
                f"loaded from {period['satellite_track_csv']!r}"
            )

        has_ascending = "ascending" in sat.columns
        for delta in start_deltas:
            forecast_start, query_start_dt = resolve_start_delta(period["start"], period["end"], delta)
            model.forecast(forecast_start, period["end"])

            mask = (sat["datetime"] >= query_start_dt).to_numpy()
            sat_slice = sat[mask].reset_index(drop=True)
            phys_slice = phys[mask].reset_index(drop=True)
            if sat_slice.empty:
                raise ValueError(
                    f"period {period['label']!r}: start_delta {delta!r}h leaves no rows in "
                    f"[{query_start_dt}, {period['end']})"
                )

            rope_values = [
                model.query(row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), row["lst"], row["lat"], row["alt_km"])[variable]
                for _, row in sat_slice.iterrows()
            ]

            for i in range(len(sat_slice)):
                row = {
                    "period": period["label"], "start_delta": delta,
                    "datetime": sat_slice["datetime"].iat[i], "lst": sat_slice["lst"].iat[i],
                    "lat": sat_slice["lat"].iat[i], "alt_km": sat_slice["alt_km"].iat[i],
                    "satellite_density": sat_slice[variable].iat[i], "physics_density": phys_slice[variable].iat[i],
                    "rope_density": rope_values[i], "orbit_averaged": bool(period.get("orbit_averaged", False)),
                    "plot_satellite_data": bool(period.get("plot_satellite_data", True)),
                }
                if has_ascending:
                    row["ascending"] = sat_slice["ascending"].iat[i]
                rows.append(row)

    comparison = pd.DataFrame(rows)
    data_path = save_csv(out_dir, f"{id}.csv", comparison)

    plots = []
    stats_by_period = {}
    per_period = {}
    for period in periods:
        label = period["label"]
        start_deltas = period.get("start_deltas", [0])
        n_deltas = len(start_deltas)
        orbit_averaged = bool(period.get("orbit_averaged", False))
        plot_satellite_data = bool(period.get("plot_satellite_data", True))
        widest_delta = min(start_deltas)

        label_rows = comparison[comparison["period"] == label]
        widest_rows = label_rows[label_rows["start_delta"] == widest_delta]
        if orbit_averaged:
            widest_rows = _orbit_average(widest_rows, label)
        sat_display = widest_rows["satellite_density"].to_numpy()
        phys_display = widest_rows["physics_density"].to_numpy()

        stats_by_delta = {}
        per_period_delta = {}
        rope_series, stats_text_series = {}, None
        for delta in start_deltas:
            delta_rows = label_rows[label_rows["start_delta"] == delta]
            if orbit_averaged:
                delta_rows = _orbit_average(delta_rows, label)

            rope_arr = delta_rows["rope_density"].to_numpy()
            sat_arr = delta_rows["satellite_density"].to_numpy()
            phys_arr = delta_rows["physics_density"].to_numpy()

            rope_vs_sat = compute_statistics(rope_arr, sat_arr, statistics)
            phys_vs_sat = compute_statistics(phys_arr, sat_arr, statistics)
            rope_vs_phys = compute_statistics(rope_arr, phys_arr, statistics)
            delta_stats = {}
            if rope_vs_sat is not None:
                delta_stats["rope_vs_satellite"] = rope_vs_sat
            if phys_vs_sat is not None:
                delta_stats["physics_vs_satellite"] = phys_vs_sat
            if rope_vs_phys is not None:
                delta_stats["rope_vs_physics_model"] = rope_vs_phys
            if delta_stats:
                stats_by_delta[delta_stat_key(delta)] = delta_stats
                stats_text_series = (rope_vs_sat, rope_vs_phys)

            rope_series[delta_label(rope_model_label, delta, n_deltas=n_deltas)] = (
                delta_rows["datetime"], rope_arr
            )

            value = float(np.sqrt(np.mean(np.square(rope_arr - sat_arr))))
            passed = passes_threshold(value, threshold) if threshold else None
            per_period_delta[delta_stat_key(delta)] = {"value": value, "passed": passed}

        if stats_by_delta:
            stats_by_period[label] = stats_by_delta
        per_period[label] = per_period_delta

        stats_text = None
        if n_deltas == 1 and stats_text_series is not None:
            rope_vs_sat_stats, rope_vs_phys_stats = stats_text_series
            parts = []
            if rope_vs_sat_stats is not None:
                parts.append("rope_vs_satellite:\n" + format_statistics_text(rope_vs_sat_stats))
            if rope_vs_phys_stats is not None:
                parts.append("rope_vs_physics:\n" + format_statistics_text(rope_vs_phys_stats))
            stats_text = "\n".join(parts) if parts else None

        panel = {
            "title": label,
            "ylabel": unit or variable,
            "series": {
                **({satellite_label: (widest_rows["datetime"], sat_display)} if plot_satellite_data else {}),
                physics_model_label: (widest_rows["datetime"], phys_display),
                **rope_series,
            },
            "stats_text": stats_text,
        }
        plot_name = f"plots/{id}_{label}.png"
        line_plot([panel], out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {label}")
        plots.append(plot_name)

    passed_by_label = {
        label: all(pp["passed"] for pp in per_delta.values()) if threshold else None
        for label, per_delta in per_period.items()
    }
    output = {
        "plots": plots, "data": [data_path],
        "passed": all(passed_by_label.values()) if threshold else None,
        "per_period": per_period,
    }
    if stats_by_period:
        output["statistics"] = stats_by_period
    return output


def replot_satellite_orbit_density(loaded: dict, *, id, out_dir, unit=None) -> list:
    """loaded: {relative_data_path: DataFrame}, as produced by generate_validation_plots.py."""
    data = loaded[f"validation_data/{id}.csv"]
    labels = list(dict.fromkeys(data["period"]))

    plots = []
    for label in labels:
        label_rows = data[data["period"] == label]
        start_deltas = sorted(label_rows["start_delta"].unique())
        n_deltas = len(start_deltas)
        orbit_averaged = "orbit_averaged" in label_rows.columns and bool(label_rows["orbit_averaged"].iat[0])
        plot_satellite_data = (
            "plot_satellite_data" not in label_rows.columns or bool(label_rows["plot_satellite_data"].iat[0])
        )
        widest_delta = min(start_deltas)

        widest_rows = label_rows[label_rows["start_delta"] == widest_delta]
        if orbit_averaged:
            widest_rows = _orbit_average(widest_rows, label)

        rope_series = {}
        for delta in start_deltas:
            delta_rows = label_rows[label_rows["start_delta"] == delta]
            if orbit_averaged:
                delta_rows = _orbit_average(delta_rows, label)
            rope_series[delta_label("rope_model", delta, n_deltas=n_deltas)] = (
                delta_rows["datetime"], delta_rows["rope_density"]
            )

        panel = {
            "title": label,
            "ylabel": unit or "density",
            "series": {
                **({"satellite": (widest_rows["datetime"], widest_rows["satellite_density"])}
                   if plot_satellite_data else {}),
                "physics_model": (widest_rows["datetime"], widest_rows["physics_density"]),
                **rope_series,
            },
        }
        plot_name = f"plots/{id}_{label}.png"
        line_plot([panel], out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {label}")
        plots.append(plot_name)
    return plots
