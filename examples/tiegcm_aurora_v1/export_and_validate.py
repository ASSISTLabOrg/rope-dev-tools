#!/usr/bin/env python3
"""Export tiegcm-aurora-v1 and run its validation suite against the exported dir.

Usage:
    ./export_and_validate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rope_dev_tools import export_model
from rope_dev_tools.spec import load_spec

EXAMPLE_DIR = Path(__file__).resolve().parent

VALIDATION_DIR = Path(os.environ.get("VALIDATION_DIR", "/path/to/tiegcm-aurora-v1-validation"))
SUITE = Path(os.environ.get("SUITE", VALIDATION_DIR / "suite.json"))
DRIVER_PATH = Path(os.environ.get("DRIVER_PATH", VALIDATION_DIR / "driver.csv"))
OUT_DIR = Path(os.environ.get("OUT_DIR", EXAMPLE_DIR / "export"))
FORCE = bool(os.environ.get("FORCE"))


def main() -> int:
    spec = load_spec(f"{EXAMPLE_DIR / 'spec.py'}:SPEC")
    result = export_model(spec, OUT_DIR, suite=SUITE, driver_path=DRIVER_PATH, force=FORCE)

    print(f"wrote manifest: {result.manifest_path}")
    if result.passed is not None:
        print(f"wrote validation report: {result.report_path}")
        if not result.passed:
            print("one or more checks failed; see the report for details", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
