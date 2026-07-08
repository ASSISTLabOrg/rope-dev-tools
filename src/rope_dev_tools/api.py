"""export_model(), verify_model(), mark_validated(), upgrade_manifest() — the
real implementations, each returning a small result dataclass. cli.py's
subcommands are thin argparse shells that parse flags and call straight into
these — a dev can call these directly from a script or notebook instead of
shelling out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rope_dev_tools.export.base import get_exporter
from rope_dev_tools.manifest import ManifestBuilder
from rope_dev_tools.registry.schema_store import RegistrySchemaStore
from rope_dev_tools.registry.validate import ManifestValidator
from rope_dev_tools.registry.vendor import RegistryVendor
from rope_dev_tools.spec import ModelSpec, load_python_attr
from rope_dev_tools.validation.model_interfaces import ExportedDirModelInterface, WrapperModelInterface
from rope_dev_tools.validation.runner import validate as run_validate
from rope_dev_tools.validation.schema_types import ValidationSuite, report_all_passed


def _default_validator() -> ManifestValidator:
    root = RegistryVendor().resolve()
    store = RegistrySchemaStore(root)
    return ManifestValidator(store)


@dataclass
class ExportResult:
    manifest: dict
    manifest_path: Path
    out_dir: Path
    report: "dict | None" = None
    report_path: "Path | None" = None


@dataclass
class VerificationResult:
    report: dict
    report_path: Path
    passed: bool


def export_model(
    spec: ModelSpec,
    out_dir: "str | Path",
    *,
    suite: "str | Path | None" = None,
    wrapper: "str | None" = None,
    package_root: "str | Path | None" = None,
    driver_path: "str | Path | None" = None,
    skip_validation: bool = False,
    skip_conversion_check: bool = False,
    force: bool = False,
    validator: "ManifestValidator | None" = None,
) -> ExportResult:
    """Converts spec.source_dir's trained artifacts into out_dir and writes a
    schema-validated model_manifest.json.

    Runs the validation suite by default unless skip_validation=True or no
    suite= is given — exported-dir mode (the real rope-framework binary)
    unless wrapper= is given. Never sets manifest['validated']; that's
    mark_validated()'s job, run separately after a human reviews the report.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"{out_dir} is not empty; pass force=True to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    validator = validator or _default_validator()

    if skip_conversion_check:
        spec.kind_params.setdefault("skip_conversion_check", True)

    exporter = get_exporter(spec.kind)
    kind_block = exporter.export(spec, out_dir)

    builder = ManifestBuilder(validator)
    manifest = builder.build_and_validate(spec, kind_block)
    manifest_path = builder.write(manifest, out_dir, validate=False)

    result = ExportResult(manifest=manifest, manifest_path=manifest_path, out_dir=out_dir)

    if suite is not None and not skip_validation:
        verification = verify_model(
            out_dir, suite, wrapper=wrapper, package_root=package_root,
            driver_path=driver_path, validator=validator,
        )
        result.report = verification.report
        result.report_path = verification.report_path

    return result


def verify_model(
    exported_dir: "str | Path",
    suite: "str | Path",
    *,
    wrapper: "str | None" = None,
    package_root: "str | Path | None" = None,
    driver_path: "str | Path | None" = None,
    validator: "ManifestValidator | None" = None,
) -> VerificationResult:
    """Runs a validation suite against exported_dir.

    Uses exported-dir mode (the real rope-framework binary/library) by
    default, or wrapper mode if wrapper= ('module_or_path.py:function') is
    given.
    """
    validator = validator or _default_validator()
    suite_path = Path(suite)
    validation_suite = ValidationSuite.from_json(suite_path)

    if wrapper is not None:
        wrapper_fn = load_python_attr(wrapper)
        model = WrapperModelInterface(wrapper_fn)
    else:
        model = ExportedDirModelInterface(exported_dir, package_root=package_root, driver_path=driver_path)

    try:
        report = run_validate(model, validation_suite, exported_dir, suite_dir=suite_path.parent, validator=validator)
    finally:
        model.close()

    return VerificationResult(
        report=report,
        report_path=Path(exported_dir) / "validation_report.json",
        passed=report_all_passed(report),
    )


def mark_validated(
    exported_dir: "str | Path",
    *,
    report_path: "str | Path | None" = None,
    validator: "ManifestValidator | None" = None,
) -> dict:
    """Its own operation, not a step of export: flips validated=true and
    fills the validation block from an already-produced report. Never
    re-converts artifacts and never re-runs the suite.
    """
    validator = validator or _default_validator()
    exported_dir = Path(exported_dir)
    report_path = Path(report_path) if report_path else exported_dir / "validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"{report_path} not found — run verify_model() / `rope-dev-tools verify` first"
        )
    report = json.loads(report_path.read_text())

    builder = ManifestBuilder(validator)
    return builder.set_validated(exported_dir, report, report_filename=report_path.name)


def upgrade_manifest(
    manifest_path: "str | Path",
    *,
    validator: "ManifestValidator | None" = None,
) -> dict:
    """Migrates an existing legacy-shape manifest (top-level ic_grid_axes, no
    'validated' field) to registry shape, in place.
    """
    validator = validator or _default_validator()
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    builder = ManifestBuilder(validator)
    upgraded = builder.upgrade_legacy(manifest, exported_dir=manifest_path.parent)
    manifest_path.write_text(json.dumps(upgraded, indent=2) + "\n")
    return upgraded
