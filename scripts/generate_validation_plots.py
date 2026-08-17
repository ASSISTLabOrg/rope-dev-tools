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
import inspect
import json
import sys
from pathlib import Path

from rope_dev_tools.validation.checks import get_replot_function
from rope_dev_tools.validation.data_artifacts import load_csv, load_npz

# Imported for their @register_replot side effect -- each module registers its own replot_<kind> function.
import rope_dev_tools.validation.checks.altitude_profile  # noqa: F401
import rope_dev_tools.validation.checks.avg_density_vs_time  # noqa: F401
import rope_dev_tools.validation.checks.latitude_profile  # noqa: F401
import rope_dev_tools.validation.checks.harmonic_fft  # noqa: F401
import rope_dev_tools.validation.checks.lonlat_snapshot_series  # noqa: F401
import rope_dev_tools.validation.checks.satellite_orbit_density  # noqa: F401


def _load_data(exported_dir: Path, relative_paths: list) -> dict:
    return {
        path: load_csv(exported_dir, path) if path.endswith(".csv") else load_npz(exported_dir, path)
        for path in relative_paths
    }


def _check_kwargs_from_suite(suite: dict, check_id: str) -> dict:
    """Extracts replot-relevant kwargs from the suite JSON for a given check id."""
    for check in suite.get("checks", []):
        if check.get("id") == check_id:
            skip = {"id", "kind", "periods", "altitudes_km", "requires_exported_model"}
            return {k: v for k, v in check.items() if k not in skip}
    return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exported-dir", required=True)
    parser.add_argument("--suite", default=None, help="suite JSON file for check-level kwargs (unit, altitude_ylim, etc.)")
    parser.add_argument("--check-id", action="append", default=None,
                         help="regenerate only these check ids (default: all)")
    args = parser.parse_args(argv)

    exported_dir = Path(args.exported_dir)
    report = json.loads((exported_dir / "validation_report.json").read_text())

    suite = json.loads(Path(args.suite).read_text()) if args.suite else {}

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

        try:
            replot_fn = get_replot_function(result["kind"])
        except KeyError:
            print(f"{result['id']!r}: unknown kind {result['kind']!r}, skipping", file=sys.stderr)
            exit_code = 1
            continue

        loaded = _load_data(exported_dir, data_paths)
        extra = _check_kwargs_from_suite(suite, result["id"])
        for key in ("physics_model_label", "rope_model_label", "satellite_label"):
            if key in suite and key not in extra:
                extra[key] = suite[key]
        # Only pass kwargs the replot function actually accepts.
        sig = inspect.signature(replot_fn)
        if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            extra = {k: v for k, v in extra.items() if k in sig.parameters}
        plots = replot_fn(loaded, id=result["id"], out_dir=exported_dir, **extra)
        print(f"{result['id']!r}: wrote {len(plots)} plot(s)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
