#!/usr/bin/env python3
"""
build_satellite_data.py — builds satellite_track_csv for a suite from the ESA swarm-diss server
(GRACE/GRACE-FO) or a local mirror.

Usage
~~~~~
  python scripts/build_satellite_data.py --suite rope-data/validation/validation-wam-v1.json \
      --out-dir rope-data/validation --source remote --source-config rope-data/validation/satellite_sources.json

  python scripts/build_satellite_data.py --suite ... --out-dir ... \
      --source offline --source-config rope-data/validation/satellite_sources.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rope_dev_tools.validation.satellite_ingest import build
from rope_dev_tools.validation.satellite_source import (
    LocalMirrorSatelliteSource,
    SwarmDissSatelliteSource,
    load_satellite_source_config,
)
from rope_dev_tools.validation.schema_types import ValidationSuite


def _print_progress(index: int, total: int, day) -> None:
    print(f"[{index + 1}/{total}] fetching {day}", flush=True)


def _make_source(source_kind: str, config_path: Path):
    config = load_satellite_source_config(config_path)
    years = config.get(source_kind, {})
    default_satellite = config["default_satellite"]

    if source_kind == "remote":
        return SwarmDissSatelliteSource(years, default_satellite=default_satellite)
    return LocalMirrorSatelliteSource(years, default_satellite=default_satellite)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, help="validation suite JSON")
    parser.add_argument("--out-dir", required=True, help="directory to write satellite_track_csv files into")
    parser.add_argument("--source", required=True, choices=["remote", "offline"])
    parser.add_argument("--source-config", required=True, help="satellite_sources.json")
    parser.add_argument("--only-check", action="append", default=None, dest="only_check_ids",
                         help="restrict to this check id (repeatable); default: every check in the suite")
    parser.add_argument("--cadence-seconds", type=int, default=600,
                         help="downsample raw 10s rows by averaging into bins this wide; 0 disables downsampling")
    parser.add_argument("--quiet", action="store_true",
                         help="suppress per-fetch progress lines (a large suite can mean thousands "
                              "of individual fetches; progress is on by default so a long run doesn't look hung)")
    args = parser.parse_args(argv)

    suite = ValidationSuite.from_json(args.suite)
    source = _make_source(args.source, Path(args.source_config))
    written = build(suite, Path(args.out_dir), source, only_check_ids=args.only_check_ids,
                     cadence_seconds=args.cadence_seconds, progress=None if args.quiet else _print_progress)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
