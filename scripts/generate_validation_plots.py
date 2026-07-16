#!/usr/bin/env python3
"""
generate_validation_plots.py — regenerates plots from a saved validation_report.json's data
artifacts, with no model/inference/rope-framework dependency.

Usage
~~~~~
  python scripts/generate_validation_plots.py --exported-dir <dir>
  python scripts/generate_validation_plots.py --exported-dir <dir> --check-id foo --check-id bar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rope_dev_tools.validation.checks.avg_density_vs_time import replot_avg_density_vs_time
from rope_dev_tools.validation.checks.doy_lat_orbit_density import replot_doy_lat_orbit_density
from rope_dev_tools.validation.checks.lonlat_snapshot_series import replot_lonlat_snapshot_series
from rope_dev_tools.validation.checks.satellite_orbit_density import replot_satellite_orbit_density
from rope_dev_tools.validation.data_artifacts import load_csv, load_npz

_REPLOT_FUNCTIONS = {
    "avg_density_vs_time": replot_avg_density_vs_time,
    "lonlat_snapshot_series": replot_lonlat_snapshot_series,
    "satellite_orbit_density": replot_satellite_orbit_density,
    "doy_lat_orbit_density": replot_doy_lat_orbit_density,
}


def _load_data(exported_dir: Path, relative_paths: list) -> dict:
    return {
        path: load_csv(exported_dir, path) if path.endswith(".csv") else load_npz(exported_dir, path)
        for path in relative_paths
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exported-dir", required=True)
    parser.add_argument("--check-id", action="append", default=None,
                         help="regenerate only these check ids (default: all)")
    args = parser.parse_args(argv)

    exported_dir = Path(args.exported_dir)
    report = json.loads((exported_dir / "validation_report.json").read_text())

    wanted = set(args.check_id) if args.check_id else None
    exit_code = 0

    for result in report["results"]:
        if wanted is not None and result["id"] not in wanted:
            continue

        data_paths = result["output"].get("data")
        if not data_paths:
            print(f"{result['id']!r} (kind {result['kind']!r}): no saved data artifact, skipping", file=sys.stderr)
            exit_code = 1
            continue

        replot_fn = _REPLOT_FUNCTIONS.get(result["kind"])
        if replot_fn is None:
            print(f"{result['id']!r}: unknown kind {result['kind']!r}, skipping", file=sys.stderr)
            exit_code = 1
            continue

        loaded = _load_data(exported_dir, data_paths)
        plots = replot_fn(loaded, id=result["id"], out_dir=exported_dir)
        print(f"{result['id']!r}: wrote {len(plots)} plot(s)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
