"""validate() — runs a ValidationSuite against a ModelInterface, producing a report."""

from __future__ import annotations

import json
from pathlib import Path

from rope_dev_tools.validation.checks import get_kind_function, passes_threshold
from rope_dev_tools.validation.schema_types import ValidationSuite, build_report


class SuiteShapeError(ValueError):
    pass


def _check_suite_shape(suite: ValidationSuite) -> None:
    ids = []
    for check in suite.checks:
        if "id" not in check or "kind" not in check:
            raise SuiteShapeError(f"check missing 'id' or 'kind': {check!r}")
        ids.append(check["id"])
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SuiteShapeError(f"duplicate check ids: {dupes}")


def validate(model, suite: ValidationSuite, out_dir: Path, *, suite_dir: Path) -> dict:
    _check_suite_shape(suite)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for check in suite.checks:
        fn = get_kind_function(check["kind"])
        fields = {k: v for k, v in check.items() if k not in ("id", "kind")}
        output = fn(model, id=check["id"], out_dir=out_dir, suite_dir=suite_dir, **fields)
        results.append({"id": check["id"], "kind": check["kind"], "output": output})

    report = build_report(suite.content_version, results)

    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def recheck_report(report: dict, suite: ValidationSuite) -> dict:
    """Re-evaluates passed/fail against the suite's current thresholds without re-running inference."""
    thresholds = {c["id"]: c.get("threshold") for c in suite.checks}
    new_results = []
    for r in report["results"]:
        threshold = thresholds.get(r["id"])
        output = r["output"]
        if threshold and isinstance(output, dict) and isinstance(output.get("value"), (int, float)):
            output = {**output, "passed": passes_threshold(output["value"], threshold)}
        new_results.append({**r, "output": output})

    return {**report, "results": new_results}
