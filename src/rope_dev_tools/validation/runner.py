"""validate() — runs a ValidationSuite against a ModelInterface, producing a report."""

from __future__ import annotations

import json
from pathlib import Path

from rope_dev_tools.validation.checks import get_kind_function, passes_threshold
from rope_dev_tools.validation.schema_types import ValidationSuite, build_report


class SuiteShapeError(ValueError):
    pass


def _check_suite_shape(suite: ValidationSuite) -> None:
    """Raises SuiteShapeError if a check is missing id/kind, or ids collide."""
    ids = []
    for check in suite.checks:
        if "id" not in check or "kind" not in check:
            raise SuiteShapeError(f"check missing 'id' or 'kind': {check!r}")
        ids.append(check["id"])
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SuiteShapeError(f"duplicate check ids: {dupes}")


def validate(model, suite: ValidationSuite, out_dir: Path, *, suite_dir: Path, only_check_ids=None,
             progress=None) -> dict:
    """Runs suite's checks against model, writing out_dir/validation_report.json."""
    _check_suite_shape(suite)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = suite.checks if only_check_ids is None else [c for c in suite.checks if c["id"] in only_check_ids]

    results = []
    for i, check in enumerate(checks):
        if progress is not None:
            progress(i, len(checks), check["id"])
        fn = get_kind_function(check["kind"])
        fields = {k: v for k, v in check.items() if k not in ("id", "kind")}
        output = fn(
            model, id=check["id"], out_dir=out_dir, suite_dir=suite_dir,
            physics_model_label=suite.physics_model_label, rope_model_label=suite.rope_model_label,
            satellite_label=suite.satellite_label, **fields,
        )
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
        if threshold and isinstance(output, dict):
            if isinstance(output.get("per_period"), dict):
                per_period = {
                    label: {
                        delta_key: {**pp, "passed": passes_threshold(pp["value"], threshold)}
                        for delta_key, pp in per_delta.items()
                    }
                    for label, per_delta in output["per_period"].items()
                }
                passed_by_label = {
                    label: all(pp["passed"] for pp in per_delta.values())
                    for label, per_delta in per_period.items()
                }
                output = {**output, "per_period": per_period, "passed": all(passed_by_label.values())}
            elif isinstance(output.get("value"), (int, float)):
                output = {**output, "passed": passes_threshold(output["value"], threshold)}
        new_results.append({**r, "output": output})

    return {**report, "results": new_results}
