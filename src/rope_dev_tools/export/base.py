"""ModelExporter — the extension point for adding support for a new model kind.

Kind dispatch is a plain dict registry (the Python analogue of
rope-framework's pipeline_registry.cpp string-switch), populated by
@register_exporter. Adding a new model kind means: add a schema to
rope-registry, add the C++ pipeline in rope-framework, and add one new
ModelExporter subclass here that composes export/common.py primitives —
see docs/adding-a-model-kind.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from rope_dev_tools.spec import ModelSpec


class SpecValidationError(ValueError):
    def __init__(self, errors: list):
        super().__init__("model spec is invalid:\n  " + "\n  ".join(errors))
        self.errors = errors


class UnknownModelKindError(KeyError):
    def __init__(self, kind: str, known: list):
        super().__init__(f"unknown model kind {kind!r}; known: {sorted(known)}")
        self.kind = kind
        self.known = known


class ModelExporter(ABC):
    kind: ClassVar[str]

    def validate_spec(self, spec: ModelSpec) -> None:
        """Pre-flight check of spec.kind_params. Default: no-op.

        Subclasses should collect every problem into one SpecValidationError
        rather than raising on the first missing key, so a dev sees the full
        list of what's wrong with their spec in one pass.
        """
        return None

    @abstractmethod
    def export(self, spec: ModelSpec, out_dir: Path) -> dict:
        """Converts spec.source_dir's trained artifacts into out_dir.

        Returns the kind-specific manifest block (the value that will be
        assigned to manifest[spec.kind]). Must fail loudly and not leave a
        manifest written on any error, including a conversion-fidelity
        mismatch.
        """
        raise NotImplementedError


_EXPORTERS: dict = {}


def register_exporter(cls):
    _EXPORTERS[cls.kind] = cls
    return cls


def get_exporter(kind: str) -> ModelExporter:
    try:
        return _EXPORTERS[kind]()
    except KeyError:
        raise UnknownModelKindError(kind, list(_EXPORTERS)) from None
