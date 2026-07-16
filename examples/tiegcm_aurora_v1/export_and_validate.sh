#!/usr/bin/env bash
# Export tiegcm-aurora-v1 and run its validation suite against the exported dir.
#
# Usage:
#   ./export_and_validate.sh
#

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"

VALIDATION_DIR="${VALIDATION_DIR:-/path/to/tiegcm-aurora-v1-validation}"
SUITE="${SUITE:-$VALIDATION_DIR/suite.json}"
DRIVER_PATH="${DRIVER_PATH:-$VALIDATION_DIR/driver.csv}"
OUT_DIR="${OUT_DIR:-$EXAMPLE_DIR/export}"

args=(
    export
    --spec "$EXAMPLE_DIR/spec.py:SPEC"
    --out-dir "$OUT_DIR"
    --suite "$SUITE"
    --driver-path "$DRIVER_PATH"
)
[ -n "${FORCE:-}" ] && args+=(--force)

rope-dev-tools "${args[@]}"
