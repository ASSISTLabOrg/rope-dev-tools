"""Resolves a local, on-disk copy of the rope-registry schema repository."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

REGISTRY_URL = "https://github.com/AssistLabOrg/rope-registry/archive/refs/tags/v0.5.tar.gz"
REGISTRY_SHA256 = "e966efe87f34f77f1a3b59f1b0a6658bf14a8e9609e12511a23fe0cfe0774a59"
REGISTRY_TAG = "v0.5"

ENV_OVERRIDE = "ROPE_REGISTRY_PATH"


class RegistryFetchError(RuntimeError):
    pass


class RegistryVendor:
    """Resolves a local rope-registry checkout: env override, else a cached fetch."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "rope-dev-tools" / "rope-registry")

    def resolve(self) -> Path:
        """ENV_OVERRIDE if set (must contain schemas/), else fetch()."""
        override = os.environ.get(ENV_OVERRIDE)
        if override:
            path = Path(override)
            if not (path / "schemas").is_dir():
                raise RegistryFetchError(
                    f"{ENV_OVERRIDE}={override!r} does not look like a rope-registry "
                    f"checkout (no schemas/ subdirectory)"
                )
            return path
        return self.fetch()

    def fetch(self, force: bool = False) -> Path:
        """Downloads, checksum-verifies, and extracts the registry tarball; cached unless force=True."""
        dest = self.cache_dir / REGISTRY_TAG
        if force and dest.exists():
            shutil.rmtree(dest)

        existing = self._find_extracted_root(dest)
        if existing is not None:
            return existing

        with urllib.request.urlopen(REGISTRY_URL) as resp:  # noqa: S310 - fixed, checksum-verified URL
            data = resp.read()

        digest = hashlib.sha256(data).hexdigest()
        if digest != REGISTRY_SHA256:
            raise RegistryFetchError(
                f"rope-registry download hash mismatch: expected {REGISTRY_SHA256}, got {digest}"
            )

        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(dest)

        root = self._find_extracted_root(dest)
        if root is None:
            raise RegistryFetchError(
                "rope-registry archive extracted but no top-level schemas/ directory found"
            )
        return root

    @staticmethod
    def _find_extracted_root(dest: Path) -> Path | None:
        """The tarball's single top-level subdir if it contains schemas/, else None."""
        if not dest.is_dir():
            return None
        subdirs = [p for p in dest.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "schemas").is_dir():
            return subdirs[0]
        return None
