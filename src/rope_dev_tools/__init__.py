"""rope_dev_tools — model export and validation tooling for ROPE.

Public API: ModelSpec, export_model, verify_model, mark_validated,
upgrade_manifest, WrapperFn/WrapperRequest/WrapperResponse,
ModelExporter/register_exporter (add a new model kind), register_kind (add a
new check kind — a plain function, see docs/adding-a-check-kind.md).
Everything else (registry/, export/common.py, export/kinds/, validation/
internals) is implementation detail, not part of this contract, and may
change shape without notice.
"""

from rope_dev_tools.api import (
    ExportResult,
    VerificationResult,
    export_model,
    mark_validated,
    upgrade_manifest,
    verify_model,
)
from rope_dev_tools.export.base import ModelExporter, register_exporter
from rope_dev_tools.spec import ModelSpec
from rope_dev_tools.validation.checks import register_kind
from rope_dev_tools.validation.model_interfaces import WrapperFn, WrapperRequest, WrapperResponse

__all__ = [
    "ModelSpec",
    "export_model", "ExportResult",
    "verify_model", "VerificationResult",
    "mark_validated",
    "upgrade_manifest",
    "WrapperFn", "WrapperRequest", "WrapperResponse",
    "ModelExporter", "register_exporter",
    "register_kind",
]

# Register built-in exporters and checks so kind/check dispatch works as soon
# as rope_dev_tools is imported, without a dev needing to import submodules.
import rope_dev_tools.export.kinds.ensemble_fusion_decoder  # noqa: E402,F401
import rope_dev_tools.validation.checks.lonlat_density_plot  # noqa: E402,F401
import rope_dev_tools.validation.checks.rmse_timeseries  # noqa: E402,F401
import rope_dev_tools.validation.checks.satellite_lineout  # noqa: E402,F401
