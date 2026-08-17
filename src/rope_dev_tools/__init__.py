"""Model export and validation tooling for ROPE."""

from rope_dev_tools.api import (
    ExportResult,
    VerificationResult,
    export_model,
    mark_validated,
    upgrade_manifest,
    verify_model,
)
from rope_dev_tools.export.base import ModelExporter, SpecValidationError, UnknownModelKindError, register_exporter
from rope_dev_tools.export.common import ConversionFidelityError
from rope_dev_tools.registry.schema_store import UnknownKindError
from rope_dev_tools.registry.validate import ManifestValidationError
from rope_dev_tools.registry.vendor import RegistryFetchError
from rope_dev_tools.spec import ModelSpec, load_spec
from rope_dev_tools.validation.checks import register_kind
from rope_dev_tools.validation.model_interfaces import WrapperFn, WrapperRequest, WrapperResponse
from rope_dev_tools.validation.schema_types import report_all_passed

__all__ = [
    "ModelSpec", "load_spec",
    "export_model", "ExportResult",
    "verify_model", "VerificationResult",
    "mark_validated",
    "upgrade_manifest",
    "report_all_passed",
    "WrapperFn", "WrapperRequest", "WrapperResponse",
    "ModelExporter", "register_exporter",
    "register_kind",
    "SpecValidationError", "UnknownModelKindError",
    "ManifestValidationError", "UnknownKindError",
    "ConversionFidelityError", "RegistryFetchError",
]

import rope_dev_tools.export.kinds.stacked_ensemble  # noqa: E402,F401
import rope_dev_tools.validation.checks.altitude_profile  # noqa: E402,F401
import rope_dev_tools.validation.checks.avg_density_vs_time  # noqa: E402,F401
import rope_dev_tools.validation.checks.latitude_profile  # noqa: E402,F401
import rope_dev_tools.validation.checks.harmonic_fft  # noqa: E402,F401
import rope_dev_tools.validation.checks.lonlat_snapshot_series  # noqa: E402,F401
import rope_dev_tools.validation.checks.satellite_orbit_density  # noqa: E402,F401
