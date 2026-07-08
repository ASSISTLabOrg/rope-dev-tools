"""ModelSpec — the dict-like model description a dev provides to export a model.

This is the entire input contract for the exporter: a model `kind` (dispatches
to a ModelExporter subclass), envelope-level fields shared by every kind, and
a free-form `kind_params` dict interpreted by that kind's exporter.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_python_attr(module_attr: str):
    """Loads an object from 'path/to/module.py:ATTR' or 'package.module:ATTR'."""
    module_ref, sep, attr = module_attr.partition(":")
    if not sep:
        raise ValueError(f"expected 'module_or_path:attr', got {module_attr!r}")

    module_path = Path(module_ref)
    if module_path.suffix == ".py" and module_path.exists():
        # A spec.py commonly imports sibling helper modules (e.g. a
        # model_defs.py with architecture/loader code) -- make its directory
        # importable the same way running it as a script would.
        parent = str(module_path.resolve().parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        module_spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load module from {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    if not hasattr(module, attr):
        raise AttributeError(f"{module_ref!r} has no attribute {attr!r}")
    return getattr(module, attr)


@dataclass
class ModelSpec:
    kind: str
    name: str
    version: str
    source_dir: Path
    latent_dim: int
    driver_columns: list[str]
    driver_source: str
    runtime_requirements: dict[str, str] = field(default_factory=dict)
    # Kind-specific parameters (static values and/or loader callables),
    # interpreted entirely by that kind's ModelExporter subclass.
    kind_params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.source_dir = Path(self.source_dir)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            kwargs["extra"] = {**kwargs.get("extra", {}), **extra}
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: "str | Path") -> "ModelSpec":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_python(cls, module_attr: str) -> "ModelSpec":
        """Loads a ModelSpec from 'path/to/spec.py:ATTR' or 'package.module:ATTR'."""
        obj = load_python_attr(module_attr)
        if not isinstance(obj, cls):
            raise TypeError(f"{module_attr} is a {type(obj).__name__}, expected ModelSpec")
        return obj


def load_spec(ref: "str | Path") -> ModelSpec:
    """Loads a ModelSpec from a '.json' file path or a 'module_or_path:attr' reference."""
    ref = str(ref)
    if ref.endswith(".json"):
        return ModelSpec.from_json(ref)
    if ":" not in ref:
        raise ValueError(
            f"expected a path to a .json file or 'module_or_path:attr', got {ref!r}"
        )
    return ModelSpec.from_python(ref)
