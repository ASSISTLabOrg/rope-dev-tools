#!/usr/bin/env python3
"""
build_validation_data.py — builds physics_avg_csv/physics_model_hourly_npz for a suite from S3 or a local WAM mirror.

Usage
~~~~~
  python scripts/build_validation_data.py --suite rope-data/validation/validation-wam-v1.json \
      --out-dir rope-data/validation --source s3 --source-config rope-data/validation/wam_sources.json

  python scripts/build_validation_data.py --suite ... --out-dir ... \
      --source offline --source-config rope-data/validation/wam_sources.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rope_dev_tools.validation.schema_types import ValidationSuite
from rope_dev_tools.validation.wam_ingest import _DEFAULT_MAX_CONCURRENT_FETCHES, build
from rope_dev_tools.validation.wam_source import LocalMirrorWamSource, S3WamSource, load_wam_source_config


def _print_progress(index: int, total: int, timestamp) -> None:
    print(f"[{index + 1}/{total}] fetching {timestamp}", flush=True)


def _make_source(source_kind: str, config_path: Path):
    config = load_wam_source_config(config_path)
    block = config.get(source_kind)
    if block is None:
        raise ValueError(f"{config_path}: no {source_kind!r} block configured")

    if source_kind == "s3":
        if not block["bucket"]:
            raise ValueError(f"{config_path}: s3.bucket is not configured")
        return S3WamSource(block["years"], bucket=block["bucket"],
                            default_filename_pattern=config["default_filename_pattern"])
    return LocalMirrorWamSource(block["years"], default_filename_pattern=config["default_filename_pattern"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, help="validation suite JSON")
    parser.add_argument("--out-dir", required=True, help="directory to write truth-data artifacts into")
    parser.add_argument("--source", required=True, choices=["s3", "offline"])
    parser.add_argument("--source-config", required=True, help="wam_sources.json")
    parser.add_argument("--only-check", action="append", default=None, dest="only_check_ids",
                         help="restrict to this check id (repeatable); default: every check in the suite")
    parser.add_argument("--max-concurrent-fetches", type=int, default=_DEFAULT_MAX_CONCURRENT_FETCHES,
                         help=f"raw timestep fetches to prefetch in parallel via a thread pool "
                              f"(default: {_DEFAULT_MAX_CONCURRENT_FETCHES}); processing itself "
                              f"stays strictly ordered, only the I/O-bound fetch is overlapped")
    parser.add_argument("--quiet", action="store_true",
                         help="suppress per-fetch progress lines (a large suite can mean thousands "
                              "of individual fetches; progress is on by default so a long run doesn't look hung)")
    args = parser.parse_args(argv)

    suite_path = Path(args.suite)
    suite = ValidationSuite.from_json(suite_path)
    source = _make_source(args.source, Path(args.source_config))
    written = build(suite, Path(args.out_dir), source, suite_dir=suite_path.parent,
                     only_check_ids=args.only_check_ids, progress=None if args.quiet else _print_progress,
                     max_concurrent_fetches=args.max_concurrent_fetches)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
