"""ManifestBuilder — assembles and validates model_manifest.json."""

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
        kind_block = dict(kind_block)  # don't mutate caller's dict
        ic_block = kind_block.pop("ic", None)
        if ic_block is None:
            raise ValueError(f"kind_block for kind '{spec.kind}' is missing required 'ic' block")
        manifest = {
            "schema_version": 1,
            "kind": spec.kind,
            "runtime_requirements": dict(spec.runtime_requirements),
            "latent_dim": spec.latent_dim,
            "drivers": {
                "source": spec.driver_source,
                "columns": [self._resolve_driver_column(c) for c in spec.driver_columns],
            },
            "grid": dict(spec.grid),
            "ic": ic_block,
            "validated": False,
        }
        manifest[spec.kind] = kind_block
        return manifest

    def _resolve_driver_column(self, entry: "str | dict") -> dict:
        """Resolves one ModelSpec.driver_columns entry to a manifest {name, description}
        dict. A bare name looks up its canonical description in driver_registry.json
        (catching typos early); a {'name', 'description'} dict is an explicit override,
        used for a raw column not yet in the canonical registry."""
        if isinstance(entry, dict):
            if "name" not in entry or "description" not in entry:
                raise ValueError(
                    f"driver column dict entry must have 'name' and 'description', got {entry!r}"
                )
            return {"name": entry["name"], "description": entry["description"]}
        if isinstance(entry, str):
            for reg_entry in self.validator.store.driver_registry():
                if reg_entry["name"] == entry:
                    return {"name": entry, "description": reg_entry["description"]}
            raise ValueError(
                f"driver column {entry!r} not found in driver_registry.json and no explicit "
                f"description was given; pass a {{'name': ..., 'description': ...}} dict instead"
            )
        raise TypeError(f"driver column entry must be a str or dict, got {type(entry).__name__}")

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

    # -- mark-validated --------------------------------------------------

    def set_validated(
        self,
        exported_dir: Path,
        report: dict,
        *,
        report_filename: str = "validation_report.json",
    ) -> dict:
        """Sets validated=true on exported_dir's manifest from a report."""
        manifest_path = Path(exported_dir) / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text())
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

    # -- legacy manifest migration ---------------------------------------

    def upgrade_legacy(self, manifest: dict, exported_dir: "Path | None" = None) -> dict:
        """Migrates a legacy-shape manifest to registry shape."""
        manifest = json.loads(json.dumps(manifest))  # deep copy
        kind = manifest["kind"]
        kind_block = manifest[kind]

        if "ic" not in manifest:
            if "ic" in kind_block:
                manifest["ic"] = kind_block.pop("ic")
            else:
                grid_axes = manifest.pop("ic_grid_axes", None)
                if grid_axes is None:
                    raise ValueError(
                        "cannot upgrade: manifest has neither a top-level 'ic_grid_axes', "
                        "a nested kind-block ic, nor a top-level ic to migrate from"
                    )
                ic_file = self._discover_ic_file(exported_dir) if exported_dir else "ic_table.icbin"
                manifest["ic"] = {
                    "kind": "ic_lookup_table",
                    "params": {"grid_axes": grid_axes, "file": ic_file},
                }
        else:
            manifest.pop("ic_grid_axes", None)
            kind_block.pop("ic", None)

        if "drivers" not in manifest:
            driver_columns = manifest.pop("driver_columns", None)
            driver_source = manifest.pop("driver_source", None)
            if driver_columns is None or driver_source is None:
                raise ValueError(
                    "cannot upgrade: manifest has neither a top-level 'drivers' block nor "
                    "legacy 'driver_columns'/'driver_source' fields to migrate from"
                )
            manifest["drivers"] = {
                "source": driver_source,
                "columns": [self._resolve_driver_column(c) for c in driver_columns],
            }
        else:
            manifest.pop("driver_columns", None)
            manifest.pop("driver_source", None)

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
