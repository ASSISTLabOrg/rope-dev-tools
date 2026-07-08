"""ManifestBuilder — assembles model_manifest.json and self-validates it
against the vendored rope-registry schemas before writing. `validated` is
always written as False; only `set_validated()` (called from the separate
`mark_validated()` operation, never automatically) ever flips it to True.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rope_dev_tools.registry.validate import ManifestValidator
from rope_dev_tools.spec import ModelSpec


class ManifestBuilder:
    def __init__(self, validator: ManifestValidator):
        self.validator = validator

    def build(self, spec: ModelSpec, kind_block: dict) -> dict:
        manifest = {
            "schema_version": 1,
            "kind": spec.kind,
            "runtime_requirements": dict(spec.runtime_requirements),
            "latent_dim": spec.latent_dim,
            "driver_columns": list(spec.driver_columns),
            "driver_source": spec.driver_source,
            "validated": False,
        }
        manifest[spec.kind] = kind_block
        return manifest

    def build_and_validate(self, spec: ModelSpec, kind_block: dict) -> dict:
        manifest = self.build(spec, kind_block)
        self.validator.validate_manifest(manifest)
        return manifest

    def write(self, manifest: dict, out_dir: Path, *, validate: bool = True) -> Path:
        if validate:
            self.validator.validate_manifest(manifest)
        path = Path(out_dir) / "model_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path

    # -- mark-validated (its own operation, not a step of export) ------

    def set_validated(
        self,
        exported_dir: Path,
        report: dict,
        *,
        report_filename: str = "validation_report.json",
    ) -> dict:
        """Reads model_manifest.json from exported_dir, sets validated=true
        and fills the validation block from an already-produced report,
        re-validates, and writes it back in place. Never re-converts
        artifacts and never re-runs the suite.
        """
        manifest_path = Path(exported_dir) / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Every check's own output, keyed by check id -- no shape is shared
        # or imposed across kinds, so this just dumps whatever each one returned.
        summary = {result["id"]: result["output"] for result in report.get("results", [])}

        manifest["validated"] = True
        manifest["validation"] = {
            "suite_content_version": report["suite_content_version"],
            "validated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "report_file": report_filename,
            "summary": summary,
        }

        self.validator.validate_manifest(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    # -- migrating a legacy-shape manifest -----------------------------

    def upgrade_legacy(self, manifest: dict, exported_dir: "Path | None" = None) -> dict:
        """Migrates a legacy-shape manifest (top-level ic_grid_axes, no
        'validated' field) to registry shape (nested ic block, validated:
        false). A manifest already in registry shape is returned unchanged
        (aside from a no-op re-validation).
        """
        manifest = json.loads(json.dumps(manifest))  # deep copy
        kind = manifest["kind"]
        kind_block = manifest[kind]

        if "ic" not in kind_block:
            grid_axes = manifest.pop("ic_grid_axes", None)
            if grid_axes is None:
                raise ValueError(
                    "cannot upgrade: manifest has neither a top-level 'ic_grid_axes' "
                    "nor a nested ic block to migrate from"
                )
            ic_file = self._discover_ic_file(exported_dir) if exported_dir else "ic_table.icbin"
            kind_block["ic"] = {
                "kind": "ic_lookup_table",
                "params": {"grid_axes": grid_axes, "file": ic_file},
            }
        else:
            manifest.pop("ic_grid_axes", None)

        manifest.setdefault("validated", False)

        self.validator.validate_manifest(manifest)
        return manifest

    @staticmethod
    def _discover_ic_file(exported_dir) -> str:
        exported_dir = Path(exported_dir)
        if (exported_dir / "ic_table.icbin").exists():
            return "ic_table.icbin"
        if (exported_dir / "ic_table.csv").exists():
            return "ic_table.csv"
        return "ic_table.icbin"
