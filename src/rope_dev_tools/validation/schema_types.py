"""ValidationSuite/ValidationReport — thin wrappers around plain dicts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ValidationSuite:
    def __init__(self, schema_version: int, content_version: int, checks: list, *,
                 physics_model_label: "str | None" = None, rope_model_label: "str | None" = None,
                 satellite_label: "str | None" = None):
        self.schema_version = schema_version
        self.content_version = content_version
        self.checks = checks
        self.physics_model_label = physics_model_label
        self.rope_model_label = rope_model_label
        self.satellite_label = satellite_label

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationSuite":
        return cls(
            schema_version=d["schema_version"],
            content_version=d["content_version"],
            checks=list(d["checks"]),
            physics_model_label=d.get("physics_model_label"),
            rope_model_label=d.get("rope_model_label"),
            satellite_label=d.get("satellite_label"),
        )

    @classmethod
    def from_json(cls, path) -> "ValidationSuite":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "content_version": self.content_version,
            "checks": list(self.checks),
        }
        for key in ("physics_model_label", "rope_model_label", "satellite_label"):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        return d


def build_report(suite_content_version: int, results: list) -> dict:
    return {
        "schema_version": 1,
        "suite_content_version": suite_content_version,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,  # list[dict], each {"id", "kind", "output"}
    }


def report_summary(report: dict) -> dict:
    """Every check's own output, keyed by check id."""
    return {r["id"]: r["output"] for r in report["results"]}


def report_all_passed(report: dict) -> bool:
    """False iff any result's output has "passed": False."""
    for r in report["results"]:
        output = r.get("output")
        if isinstance(output, dict) and output.get("passed") is False:
            return False
    return True
