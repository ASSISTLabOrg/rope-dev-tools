"""The shared, multi-stage resolve-and-validate algorithm for rope-registry documents."""

from __future__ import annotations

from jsonschema import Draft202012Validator

from rope_dev_tools.registry.schema_store import RegistrySchemaStore


class ManifestValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("manifest failed schema validation:\n  " + "\n  ".join(errors))
        self.errors = errors


class SuiteValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("validation suite failed validation:\n  " + "\n  ".join(errors))
        self.errors = errors


class ReportValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("validation report failed schema validation:\n  " + "\n  ".join(errors))
        self.errors = errors


def _schema_errors(schema: dict, instance, *, at: str) -> list[str]:
    validator = Draft202012Validator(schema)
    return [f"{at}: {e.message} (path: {'/'.join(str(p) for p in e.absolute_path)})"
            for e in validator.iter_errors(instance)]


class ManifestValidator:
    def __init__(self, schema_store: RegistrySchemaStore):
        self.store = schema_store

    # -- manifests ---------------------------------------------------

    def validate_manifest(self, manifest: dict) -> None:
        errors = list(_schema_errors(self.store.envelope_schema(), manifest, at="envelope"))

        kind = manifest.get("kind")
        kind_block = manifest.get(kind) if kind else None
        if kind and kind_block is not None:
            try:
                kind_schema = self.store.kind_schema(kind)
            except KeyError as e:
                errors.append(f"kind: {e}")
                kind_schema = None
            if kind_schema is not None:
                errors.extend(_schema_errors(kind_schema, kind_block, at=f"{kind}"))

                ic_block = kind_block.get("ic") if isinstance(kind_block, dict) else None
                if isinstance(ic_block, dict) and "kind" in ic_block:
                    try:
                        ic_schema = self.store.ic_schema(ic_block["kind"])
                    except KeyError as e:
                        errors.append(f"ic: {e}")
                        ic_schema = None
                    if ic_schema is not None:
                        errors.extend(
                            _schema_errors(ic_schema, ic_block.get("params", {}),
                                           at=f"{kind}.ic.params")
                        )

        if errors:
            raise ManifestValidationError(errors)

    # -- validation suites ---------------------------------------------

    def validate_suite(self, suite: dict) -> None:
        errors = list(_schema_errors(self.store.suite_schema(), suite, at="suite"))

        check_kinds = self.store.check_kinds()
        known_check_kinds = {entry["kind"] for entry in check_kinds}

        for check in suite.get("checks", []):
            kind = check.get("kind")
            if kind in known_check_kinds:
                check_schema = self.store.check_schema(kind)
                fields = {k: v for k, v in check.items() if k not in ("id", "kind")}
                errors.extend(
                    _schema_errors(check_schema, fields, at=f"checks[{check.get('id')}]")
                )

        errors.extend(self._cross_check_errors(suite, known_check_kinds))

        if errors:
            raise SuiteValidationError(errors)

    @staticmethod
    def _cross_check_errors(suite: dict, known_check_kinds: set) -> list[str]:
        """Whole-document rules plain JSON Schema can't express."""
        errors: list[str] = []
        checks = suite.get("checks", [])

        check_ids = [c["id"] for c in checks]
        if len(check_ids) != len(set(check_ids)):
            dupes = sorted({c for c in check_ids if check_ids.count(c) > 1})
            errors.append(f"duplicate check ids: {dupes}")

        for check in checks:
            if check.get("kind") not in known_check_kinds:
                errors.append(f"check {check.get('id')!r} has unknown check kind {check.get('kind')!r}")

        return errors

    # -- validation reports ---------------------------------------------

    def validate_report(self, report: dict) -> None:
        errors = _schema_errors(self.store.report_schema(), report, at="report")
        if errors:
            raise ReportValidationError(errors)
