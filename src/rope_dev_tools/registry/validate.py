"""The shared, multi-stage resolve-and-validate algorithm for rope-registry manifest documents."""

from __future__ import annotations

from jsonschema import Draft202012Validator

from rope_dev_tools.registry.schema_store import RegistrySchemaStore


class ManifestValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("manifest failed schema validation:\n  " + "\n  ".join(errors))
        self.errors = errors


def _schema_errors(schema: dict, instance, *, at: str) -> list[str]:
    validator = Draft202012Validator(schema)
    return [f"{at}: {e.message} (path: {'/'.join(str(p) for p in e.absolute_path)})"
            for e in validator.iter_errors(instance)]


class ManifestValidator:
    def __init__(self, schema_store: RegistrySchemaStore):
        self.store = schema_store

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

        ic_block = manifest.get("ic")
        if isinstance(ic_block, dict) and "kind" in ic_block:
            try:
                ic_schema = self.store.ic_schema(ic_block["kind"])
            except KeyError as e:
                errors.append(f"ic: {e}")
                ic_schema = None
            if ic_schema is not None:
                errors.extend(
                    _schema_errors(ic_schema, ic_block.get("params", {}), at="ic.params")
                )

        if errors:
            raise ManifestValidationError(errors)
