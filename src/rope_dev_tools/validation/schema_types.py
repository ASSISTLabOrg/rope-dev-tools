"""ValidationSuite/ValidationReport — thin wrappers around plain dicts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ValidationSuite:
    def __init__(self, schema_version: int, content_version: int, checks: list):
        self.schema_version = schema_version
        self.content_version = content_version
        self.checks = checks  # list[dict], each at least {"id", "kind"}

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationSuite":
        return cls(
            schema_version=d["schema_version"],
            content_version=d["content_version"],
            checks=list(d["checks"]),
        )

    @classmethod
    def from_json(cls, path) -> "ValidationSuite":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "content_version": self.content_version,
            "checks": list(self.checks),
        }


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
