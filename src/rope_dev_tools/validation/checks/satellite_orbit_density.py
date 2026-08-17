"""satellite_orbit_density — satellite/physics/rope density along a satellite track, one line plot per period, any number of periods; optionally, ascending/descending doy-lat bias maps too, reusing the same forecast/query data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rope_dev_tools.validation.checks import (
    delta_label,
    delta_stat_key,
    passes_threshold,
    register_kind,
    register_replot,
)
from rope_dev_tools.validation.data_artifacts import save_csv
from rope_dev_tools.validation.plots import line_plot, lonlat_plot
from rope_dev_tools.validation.statistics import (
    compute_statistic_uncertainties,
    compute_statistics,
    format_statistics_text,
    uncertainty_of_mean,
)
from rope_dev_tools.validation.time_utils import parse_time, resolve_path, resolve_start_delta
from rope_dev_tools.validation.truth_data import load_truth_csv

_DENSITY_COLUMNS = ("satellite_density", "physics_density", "rope_density")
_DOY_LAT_DIRECTIONS = ("ascending", "descending")


def _row_ascending(df: "pd.DataFrame", label: str) -> "np.ndarray":
    """Per-row ascending/descending boolean: the 'ascending' column if fully populated, else derived from 'lat'."""
    if "ascending" in df.columns and df["ascending"].notna().all():
        return df["ascending"].astype(bool).to_numpy()
    if "lat" not in df.columns:
        raise ValueError(
            f"period {label!r}: requires an 'ascending' column, or a 'lat' column to derive "
            f"orbit direction from, in the satellite track CSV"
        )
    if len(df) < 2:
        raise ValueError(f"period {label!r}: deriving orbit direction from latitude needs at least 2 rows")
    lat = df["lat"].to_numpy(dtype=float)
    d_lat = np.diff(lat)
    return np.concatenate([d_lat >= 0, [d_lat[-1] >= 0]])


def _binned_mean(x, y, values, x_edges, y_edges) -> "np.ndarray":
    """Mean of values per (x, y) bin; NaN for empty bins."""
    total, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _abs_bias_grid(rope_grid: "np.ndarray", phys_grid: "np.ndarray") -> "np.ndarray":
    """|rope - physics| / physics * 100, NaN where physics is zero or the result is non-finite."""
    bias = 100.0 * np.abs(rope_grid - phys_grid) / np.where(phys_grid != 0, phys_grid, np.nan)
    bias[~np.isfinite(bias)] = np.nan
    return bias


def _shared_color_range(grids: list) -> tuple:
    """(min, max) across every grid, ignoring NaNs; (None, None) if nothing finite."""
    stacked = np.concatenate([np.asarray(g).ravel() for g in grids])
    if not np.isfinite(stacked).any():
        return None, None
    return float(np.nanmin(stacked)), float(np.nanmax(stacked))


def _doy_lat_plots(rows: "pd.DataFrame", *, id, label, doy_lat_bin_deg, physics_model_label,
                    rope_model_label, out_dir) -> list:
    """Ascending + descending 3-panel (physics, rope, |bias| %) day-of-year x latitude maps, one period."""
    doy = rows["datetime"].dt.dayofyear.to_numpy()
    lat = rows["lat"].to_numpy(dtype=float)
    phys = rows["physics_density"].to_numpy(dtype=float)
    rope = rows["rope_density"].to_numpy(dtype=float)
    ascending = _row_ascending(rows, label)
    doy_edges = np.arange(doy.min(), doy.max() + 2)
    lat_edges = np.arange(lat.min(), lat.max() + doy_lat_bin_deg, doy_lat_bin_deg)
    x_range = (float(doy_edges[0]), float(doy_edges[-1]))
    lat_range = (float(lat_edges[0]), float(lat_edges[-1]))

    plots = []
    for direction, mask in zip(_DOY_LAT_DIRECTIONS, (ascending, ~ascending)):
        phys_grid = _binned_mean(doy[mask], lat[mask], phys[mask], doy_edges, lat_edges)
        rope_grid = _binned_mean(doy[mask], lat[mask], rope[mask], doy_edges, lat_edges)
        bias_grid = _abs_bias_grid(rope_grid, phys_grid)
        vmin, vmax = _shared_color_range([phys_grid, rope_grid])
        _, bias_vmax = _shared_color_range([bias_grid])

        plot_name = f"plots/{id}_{label}_doy_lat_{direction}.png"
        lonlat_plot(
            [{"title": physics_model_label, "grid": phys_grid},
             {"title": rope_model_label, "grid": rope_grid},
             {"title": "|bias| %", "grid": bias_grid, "cmap": "plasma", "vmin": 0.0,
              "vmax": bias_vmax if bias_vmax is not None else 1.0, "colorbar_label": "|bias| %"}],
            n_rows=1, n_cols=3, lat_range=lat_range, x_range=x_range, xlabel="day of year",
            vmin=vmin, vmax=vmax, out_path=f"{out_dir}/{plot_name}",
            suptitle=f"{id} — {label} ({direction})",
        )
        plots.append(plot_name)
    return plots


def _orbit_average(period_rows: "pd.DataFrame", label: str, *, extra_agg: "dict | None" = None) -> "pd.DataFrame":
    """Collapses each complete ascending+descending orbit to one row, mean datetime/density (+ extra_agg columns)."""
    df = period_rows.reset_index(drop=True)
    ascending = _row_ascending(df, label)

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

    agg = {"datetime": "mean", **{col: "mean" for col in _DENSITY_COLUMNS}, **(extra_agg or {})}
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
    """Per period/start_delta: satellite/physics/rope density along the track, one line plot per period; optionally, one ascending + one descending doy-lat 3-panel map per period, from the widest start_delta's data."""
    if not periods:
        raise ValueError(f"check {id!r}: periods is empty")

    physics_model_label = physics_model_label or "physics_model"
    rope_model_label = rope_model_label or "rope_model"
    satellite_label = satellite_label or "satellite"

    rows = []
    for period in periods:
        uncertainty = period.get("uncertainty", False)
        plot_uncertainty = period.get("plot_uncertainty", False)
        if plot_uncertainty and not uncertainty:
            raise ValueError(
                f"check {id!r} period {period['label']!r} sets plot_uncertainty=true but "
                f"uncertainty=false; nothing to plot"
            )
        if period.get("plot_doy_lat", False) and period.get("doy_lat_bin_deg") is None:
            raise ValueError(
                f"check {id!r} period {period['label']!r}: plot_doy_lat=true requires doy_lat_bin_deg"
            )
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
            model.forecast(forecast_start, period["end"], compute_uncertainty=uncertainty)

            mask = (sat["datetime"] >= query_start_dt).to_numpy()
            sat_slice = sat[mask].reset_index(drop=True)
            phys_slice = phys[mask].reset_index(drop=True)
            if sat_slice.empty:
                raise ValueError(
                    f"period {period['label']!r}: start_delta {delta!r}h leaves no rows in "
                    f"[{query_start_dt}, {period['end']})"
                )

            rope_results = [
                model.query(row["datetime"].strftime("%Y-%m-%d %H:%M:%S"), row["lst"], row["lat"], row["alt_km"])
                for _, row in sat_slice.iterrows()
            ]

            for i in range(len(sat_slice)):
                row = {
                    "period": period["label"], "start_delta": delta,
                    "datetime": sat_slice["datetime"].iat[i], "lst": sat_slice["lst"].iat[i],
                    "lat": sat_slice["lat"].iat[i], "alt_km": sat_slice["alt_km"].iat[i],
                    "satellite_density": sat_slice[variable].iat[i], "physics_density": phys_slice[variable].iat[i],
                    "rope_density": rope_results[i][variable],
                    "rope_uncert": rope_results[i]["uncertainty"] if uncertainty else None,
                    "orbit_averaged": bool(period.get("orbit_averaged", False)),
                    "plot_satellite_data": bool(period.get("plot_satellite_data", True)),
                    "plot_doy_lat": bool(period.get("plot_doy_lat", False)),
                    "doy_lat_bin_deg": period.get("doy_lat_bin_deg"),
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
        uncertainty = period.get("uncertainty", False)
        plot_uncertainty = period.get("plot_uncertainty", False)
        start_deltas = period.get("start_deltas", [0])
        n_deltas = len(start_deltas)
        orbit_averaged = bool(period.get("orbit_averaged", False))
        plot_satellite_data = bool(period.get("plot_satellite_data", True))
        widest_delta = min(start_deltas)

        label_rows = comparison[comparison["period"] == label]
        widest_rows_raw = label_rows[label_rows["start_delta"] == widest_delta]
        widest_rows = _orbit_average(widest_rows_raw, label) if orbit_averaged else widest_rows_raw
        sat_display = widest_rows["satellite_density"].to_numpy()
        phys_display = widest_rows["physics_density"].to_numpy()

        stats_by_delta = {}
        per_period_delta = {}
        rope_series, stats_lines = {}, []
        for delta in start_deltas:
            delta_rows = label_rows[label_rows["start_delta"] == delta]
            if orbit_averaged:
                extra_agg = {"rope_uncert": lambda s: uncertainty_of_mean(s.to_numpy())} if uncertainty else None
                delta_rows = _orbit_average(delta_rows, label, extra_agg=extra_agg)

            rope_arr = delta_rows["rope_density"].to_numpy()
            sat_arr = delta_rows["satellite_density"].to_numpy()
            phys_arr = delta_rows["physics_density"].to_numpy()
            rope_uncert_arr = delta_rows["rope_uncert"].to_numpy() if uncertainty else None

            rope_vs_sat = compute_statistics(rope_arr, sat_arr, statistics)
            phys_vs_sat = compute_statistics(phys_arr, sat_arr, statistics)
            rope_vs_phys = compute_statistics(rope_arr, phys_arr, statistics)
            # uncertainty only applies to comparisons involving rope's own forecast -- satellite and
            # the physics model have no per-point model uncertainty of their own to propagate.
            rope_vs_sat_uncert = (
                compute_statistic_uncertainties(rope_arr, sat_arr, rope_uncert_arr, statistics)
                if uncertainty and rope_vs_sat is not None else None
            )
            rope_vs_phys_uncert = (
                compute_statistic_uncertainties(rope_arr, phys_arr, rope_uncert_arr, statistics)
                if uncertainty and rope_vs_phys is not None else None
            )
            delta_stats = {}
            if rope_vs_sat is not None:
                delta_stats["rope_vs_satellite"] = rope_vs_sat
                if rope_vs_sat_uncert:
                    delta_stats["rope_vs_satellite_uncertainty"] = rope_vs_sat_uncert
            if phys_vs_sat is not None:
                delta_stats["physics_vs_satellite"] = phys_vs_sat
            if rope_vs_phys is not None:
                delta_stats["rope_vs_physics_model"] = rope_vs_phys
                if rope_vs_phys_uncert:
                    delta_stats["rope_vs_physics_model_uncertainty"] = rope_vs_phys_uncert
            if delta_stats:
                stats_by_delta[delta_stat_key(delta)] = delta_stats
                if rope_vs_phys is not None:
                    text = format_statistics_text(rope_vs_phys, rope_vs_phys_uncert)
                    if n_deltas > 1:
                        text = "\n".join(f"Δ{delta:+d}h {line}" for line in text.split("\n"))
                    stats_lines.append(text)

            rope_label = delta_label(rope_model_label, delta, n_deltas=n_deltas)
            rope_series[rope_label] = (
                (delta_rows["datetime"], rope_arr, rope_uncert_arr) if plot_uncertainty
                else (delta_rows["datetime"], rope_arr)
            )

            value = float(np.sqrt(np.mean(np.square(rope_arr - sat_arr))))
            passed = passes_threshold(value, threshold) if threshold else None
            per_period_delta[delta_stat_key(delta)] = {"value": value, "passed": passed}

        if stats_by_delta:
            stats_by_period[label] = stats_by_delta
        per_period[label] = per_period_delta

        stats_text = "\n".join(stats_lines) if stats_lines else None

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

        if period.get("plot_doy_lat", False):
            plots += _doy_lat_plots(
                widest_rows_raw, id=id, label=label, doy_lat_bin_deg=period["doy_lat_bin_deg"],
                physics_model_label=physics_model_label, rope_model_label=rope_model_label, out_dir=out_dir,
            )

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


@register_replot("satellite_orbit_density")
def replot_satellite_orbit_density(loaded: dict, *, id, out_dir, unit=None, statistics=("bias", "rmse", "std")) -> list:
    """loaded: {relative_data_path: DataFrame}, as produced by generate_validation_plots.py."""
    data = loaded[f"validation_data/{id}.csv"]
    labels = list(dict.fromkeys(data["period"]))
    has_uncert = "rope_uncert" in data.columns and data["rope_uncert"].notna().any()

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

        widest_rows_raw = label_rows[label_rows["start_delta"] == widest_delta]
        widest_rows = _orbit_average(widest_rows_raw, label) if orbit_averaged else widest_rows_raw

        rope_series = {}
        stats_lines = []
        for delta in start_deltas:
            delta_rows = label_rows[label_rows["start_delta"] == delta]
            if orbit_averaged:
                extra_agg = {"rope_uncert": lambda s: uncertainty_of_mean(s.to_numpy())} if has_uncert else None
                delta_rows = _orbit_average(delta_rows, label, extra_agg=extra_agg)
            rope_label = delta_label("rope_model", delta, n_deltas=n_deltas)
            rope_series[rope_label] = (
                (delta_rows["datetime"], delta_rows["rope_density"], delta_rows["rope_uncert"]) if has_uncert
                else (delta_rows["datetime"], delta_rows["rope_density"])
            )
            rope_vs_phys = compute_statistics(
                delta_rows["rope_density"].to_numpy(), delta_rows["physics_density"].to_numpy(), statistics,
            )
            if rope_vs_phys is not None:
                text = format_statistics_text(rope_vs_phys)
                if n_deltas > 1:
                    text = "\n".join(f"Δ{delta:+d}h {line}" for line in text.split("\n"))
                stats_lines.append(text)

        panel = {
            "title": label,
            "ylabel": unit or "density",
            "series": {
                **({"satellite": (widest_rows["datetime"], widest_rows["satellite_density"])}
                   if plot_satellite_data else {}),
                "physics_model": (widest_rows["datetime"], widest_rows["physics_density"]),
                **rope_series,
            },
            "stats_text": "\n".join(stats_lines) if stats_lines else None,
        }
        plot_name = f"plots/{id}_{label}.png"
        line_plot([panel], out_path=f"{out_dir}/{plot_name}", suptitle=f"{id} — {label}")
        plots.append(plot_name)

        plot_doy_lat = "plot_doy_lat" in label_rows.columns and bool(label_rows["plot_doy_lat"].iat[0])
        if plot_doy_lat:
            plots += _doy_lat_plots(
                widest_rows_raw, id=id, label=label, doy_lat_bin_deg=float(label_rows["doy_lat_bin_deg"].iat[0]),
                physics_model_label="physics_model", rope_model_label="rope_model", out_dir=out_dir,
            )
    return plots
