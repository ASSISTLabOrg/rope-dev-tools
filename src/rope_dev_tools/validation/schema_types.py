"""ValidationSuite/ValidationReport — thin wrappers around plain dicts.

Checks and results are plain dicts, not dataclasses: nothing about their
shape is shared or enforced beyond {id, kind} (a check) / {id, kind, output}
(a result) — every kind's own fields/output are free-form, so there's no
common shape worth a dataclass.
"""

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
    """Every check's own output, keyed by check id -- the manifest
    validation.summary convention. No shape is imposed on output."""
    return {r["id"]: r["output"] for r in report["results"]}


def report_all_passed(report: dict) -> bool:
    """A dict output containing "passed": False is treated as a check
    failure; anything else (no "passed" key, non-dict output) is
    informational only and doesn't affect this."""
    for r in report["results"]:
        output = r.get("output")
        if isinstance(output, dict) and output.get("passed") is False:
            return False
    return True
