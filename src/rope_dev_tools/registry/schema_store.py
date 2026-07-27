"""Loads rope-registry's kind indexes and JSON Schemas from a resolved checkout."""

from __future__ import annotations

import json
from pathlib import Path


class UnknownKindError(KeyError):
    def __init__(self, family: str, name: str, known: list[str]):
        super().__init__(
            f"unknown {family} kind {name!r}; known: {sorted(known)}"
        )
        self.family = family
        self.name = name
        self.known = known


class RegistrySchemaStore:
    """Resolves kind/schema lookups against a rope-registry checkout root."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _load_json(self, relative: str) -> dict:
        path = self.root / relative
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    # -- kind indexes ----------------------------------------------------

    def pipeline_kinds(self) -> list[dict]:
        return self._load_json("pipeline_kinds.json")

    def ic_kinds(self) -> list[dict]:
        return self._load_json("ic_kinds.json")

    def driver_registry(self) -> list[dict]:
        """Canonical driver-name metadata: [{name, kind: 'raw'|'derived', description}, ...].
        Descriptive only -- not a draft/stable kind index, so unlike pipeline_kinds()/
        ic_kinds() there's no is_stable() check for it."""
        return self._load_json("driver_registry.json")

    # -- schemas -----------------------------------------------------

    def envelope_schema(self) -> dict:
        return self._load_json("schemas/manifest-envelope.schema.json")

    def kind_schema(self, kind: str) -> dict:
        return self._resolve("kind", kind, self.pipeline_kinds())

    def ic_schema(self, ic_kind: str) -> dict:
        return self._resolve("ic", ic_kind, self.ic_kinds())

    def is_stable(self, family: str, name: str) -> bool:
        registries = {
            "kind": self.pipeline_kinds,
            "ic": self.ic_kinds,
        }
        entries = registries[family]()
        for entry in entries:
            if entry["kind"] == name:
                return entry["status"] == "stable"
        raise UnknownKindError(family, name, [e["kind"] for e in entries])

    def _resolve(self, family: str, name: str, entries: list[dict]) -> dict:
        for entry in entries:
            if entry["kind"] == name:
                return self._load_json(entry["schema"])
        raise UnknownKindError(family, name, [e["kind"] for e in entries])
