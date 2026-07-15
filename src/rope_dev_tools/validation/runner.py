"""validate() — runs a ValidationSuite against a ModelInterface, producing a schema-validated report."""

from __future__ import annotations

import json
from pathlib import Path

from rope_dev_tools.registry.validate import ManifestValidator
from rope_dev_tools.validation.checks import get_kind_function, passes_threshold
from rope_dev_tools.validation.schema_types import ValidationSuite, build_report


def validate(model, suite: ValidationSuite, out_dir: Path, *, suite_dir: Path, validator: ManifestValidator) -> dict:
    validator.validate_suite(suite.to_dict())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for check in suite.checks:
        fn = get_kind_function(check["kind"])
        fields = {k: v for k, v in check.items() if k not in ("id", "kind")}
        output = fn(model, id=check["id"], out_dir=out_dir, suite_dir=suite_dir, **fields)
        results.append({"id": check["id"], "kind": check["kind"], "output": output})

    report = build_report(suite.content_version, results)
    validator.validate_report(report)

    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def recheck_report(report: dict, suite: ValidationSuite, *, validator: ManifestValidator) -> dict:
    """Re-evaluates passed/fail against the suite's current thresholds without re-running inference."""
    thresholds = {c["id"]: c.get("threshold") for c in suite.checks}
    new_results = []
    for r in report["results"]:
        threshold = thresholds.get(r["id"])
        output = r["output"]
        if threshold and isinstance(output, dict) and isinstance(output.get("value"), (int, float)):
            output = {**output, "passed": passes_threshold(output["value"], threshold)}
        new_results.append({**r, "output": output})

    new_report = {**report, "results": new_results}
    validator.validate_report(new_report)
    return new_report
