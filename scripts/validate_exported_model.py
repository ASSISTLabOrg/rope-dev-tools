#!/usr/bin/env python3
"""
validate_exported_model.py — runs a validation suite against an already-exported model directory
(bare validation: no ingestion, assumes the suite's truth-data artifacts already exist).

Usage
~~~~~
  python scripts/validate_exported_model.py --exported-dir rope-data/models/tiegcm-aurora-v1 \
      --suite rope-data/validation/validation-wam-v1.json \
      --driver-path data/drivers/sw_celestrack_1957.csv

  python scripts/validate_exported_model.py --exported-dir ... --suite ... --driver-path ... \
      --only-check avg_density_storms --only-check satellite_orbit_density
"""

from __future__ import annotations

import argparse
import sys

from rope_dev_tools import api


def _print_progress(index: int, total: int, check_id: str) -> None:
    print(f"[{index + 1}/{total}] running check {check_id!r}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exported-dir", required=True, help="directory produced by `rope-dev-tools export`")
    parser.add_argument("--suite", required=True, help="validation suite JSON")
    parser.add_argument("--package-root", default=None, help="rope-framework checkout; auto-detected if omitted")
    parser.add_argument("--build-dir", default=None,
                         help="directory containing the built rope binary/library (flat, or with "
                              "bin/+lib/ subdirs, e.g. an extracted release tarball); overrides "
                              "package-root's layout guessing (also settable via ROPE_BUILD_DIR)")
    parser.add_argument("--driver-path", default=None, help="local space-weather driver CSV/.swbin")
    parser.add_argument("--only-check", action="append", default=None, dest="only_check_ids",
                         help="restrict to this check id (repeatable); default: every check in the suite")
    parser.add_argument("--quiet", action="store_true",
                         help="suppress per-check progress lines (progress is on by default so a "
                              "suite with several checks, each of which can run a real forecast, "
                              "doesn't look hung)")
    args = parser.parse_args(argv)

    result = api.verify_model(
        args.exported_dir, args.suite,
        package_root=args.package_root, build_dir=args.build_dir, driver_path=args.driver_path,
        only_check_ids=args.only_check_ids, progress=None if args.quiet else _print_progress,
    )

    print(f"wrote validation report: {result.report_path}")
    if not result.passed:
        print("one or more checks failed; see the report for details", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
