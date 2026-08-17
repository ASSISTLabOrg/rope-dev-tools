#!/usr/bin/env python3
"""
solar_cycle_plot.py — stitches yearly avg_density_vs_time CSVs into a single
solar-cycle plot per altitude, with year x-axis labels.

Usage
~~~~~
  # Single run containing all yearly periods:
  python scripts/solar_cycle_plot.py --exported-dir runs/full -o solar_cycle.png

  # Multiple runs, one per year:
  python scripts/solar_cycle_plot.py \
      --exported-dir runs/2009 --exported-dir runs/2014 --exported-dir runs/2019 \
      -o solar_cycle.png

  # Pick specific periods and a specific start_delta:
  python scripts/solar_cycle_plot.py --exported-dir runs/full \
      --period 2009 --period 2014 --period 2019 \
      --start-delta -48 -o solar_cycle.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from rope_dev_tools.validation.statistics import compute_statistics, format_statistics_text


_CHECK_ID_DEFAULT = "Average Density"
_STATISTICS = ["bias", "rmse", "std"]
_DPI = 150


def _load_and_concat(exported_dirs: list[Path], check_id: str) -> pd.DataFrame:
    """Loads and concatenates the check's CSV from each exported dir."""
    frames = []
    for d in exported_dirs:
        csv_path = d / "validation_data" / f"{check_id}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"{csv_path} does not exist")
        frames.append(pd.read_csv(csv_path, parse_dates=["datetime"]))
    return pd.concat(frames, ignore_index=True)


def _year_formatter():
    """Tick formatter showing 'YYYY'."""
    def _format(value, _pos=None):
        return mdates.num2date(value).strftime("%Y")
    return mticker.FuncFormatter(_format)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exported-dir", action="append", required=True,
                        help="validation output directory (repeatable)")
    parser.add_argument("--check-id", default=_CHECK_ID_DEFAULT,
                        help=f"check id to load (default: {_CHECK_ID_DEFAULT!r})")
    parser.add_argument("--period", action="append", default=None,
                        help="include only these period labels (default: all)")
    parser.add_argument("--start-delta", type=int, default=None,
                        help="plot only this start_delta value (default: widest per period)")
    parser.add_argument("-o", "--out", required=True, help="output PNG path")
    parser.add_argument("--unit", default="kg/m3", help="y-axis label (default: kg/m3)")
    parser.add_argument("--truth-label", default="WAM")
    parser.add_argument("--model-label", default="ROPE")
    args = parser.parse_args(argv)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_and_concat([Path(d) for d in args.exported_dir], args.check_id)

    if args.period:
        data = data[data["period"].isin(args.period)]
        if data.empty:
            print(f"no rows match periods {args.period!r}", file=sys.stderr)
            return 1

    if args.start_delta is not None:
        data = data[data["start_delta"] == args.start_delta]
    else:
        # Keep only the widest (most-negative) start_delta per period.
        widest = data.groupby("period")["start_delta"].min()
        data = data.merge(widest.rename("_widest"), on="period")
        data = data[data["start_delta"] == data["_widest"]].drop(columns=["_widest"])

    if data.empty:
        print("no rows after filtering", file=sys.stderr)
        return 1

    data = data.sort_values("datetime")

    has_uncert = "model_uncert" in data.columns and data["model_uncert"].notna().any()
    altitudes_km = sorted(data["alt_km"].unique(), reverse=True)
    n = len(altitudes_km)

    fig, axes = plt.subplots(n, 1, figsize=(12, 4.0 * n), squeeze=False)

    for ax, alt_km in zip(axes[:, 0], altitudes_km):
        rows = data[data["alt_km"] == alt_km]
        ax.plot(rows["datetime"], rows["truth_density"],
                label=args.truth_label, linewidth=1.8)
        line, = ax.plot(rows["datetime"], rows["model_density"],
                        label=args.model_label, linewidth=1.8)
        if has_uncert:
            y = rows["model_density"].to_numpy()
            u = rows["model_uncert"].to_numpy()
            ax.fill_between(rows["datetime"], y - u, y + u,
                            color=line.get_color(), alpha=0.2, linewidth=0)

        stats = compute_statistics(
            rows["model_density"].to_numpy(), rows["truth_density"].to_numpy(), _STATISTICS,
        )
        stats_text = format_statistics_text(stats)

        ax.set_title(f"{alt_km} km", fontsize=15)
        ax.set_ylabel(args.unit, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(_year_formatter())
        if stats_text:
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va="top", fontsize=11,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7})

    axes[-1, 0].set_xlabel("year", fontsize=13)
    if n > 1:
        fig.suptitle(f"{args.check_id} — Solar Cycle", fontsize=16)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
