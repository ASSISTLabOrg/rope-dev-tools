#!/usr/bin/env python3
"""
build_validation_data.py — converts raw 4D physics-model forecast output into the truth-data
artifacts consumed by validation check kinds: physics_avg_csv (avg_density_vs_time),
physics_model_track_csv (satellite_orbit_density, doy_lat_orbit_density), physics_model_hourly_npz
(lonlat_snapshot_series).

Not yet implemented — stub only.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw 4D physics-model forecast data")
    parser.add_argument("--out-dir", required=True, help="directory to write validation-data artifacts into")
    parser.parse_args(argv)
    raise NotImplementedError("physics-data preprocessing not yet implemented")


if __name__ == "__main__":
    sys.exit(main())
