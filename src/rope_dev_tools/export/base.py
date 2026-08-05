"""ModelExporter — the extension point for adding support for a new model kind."""

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
        """Pre-flight check of spec.kind_params. Default: no-op."""
        return None

    @abstractmethod
    def export(self, spec: ModelSpec, out_dir: Path) -> dict:
        """Converts spec.source_dir's trained artifacts into out_dir. Returns the kind-specific manifest block; must fail loudly, no partial manifest."""
        raise NotImplementedError


_EXPORTERS: dict = {}


def register_exporter(cls):
    """Class decorator: registers cls under its own cls.kind in _EXPORTERS."""
    _EXPORTERS[cls.kind] = cls
    return cls


def get_exporter(kind: str) -> ModelExporter:
    """Raises UnknownModelKindError if kind isn't registered."""
    try:
        return _EXPORTERS[kind]()
    except KeyError:
        raise UnknownModelKindError(kind, list(_EXPORTERS)) from None
