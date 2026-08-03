"""model_interfaces: build_dir/package_root resolution for the exported-dir backend.

ExportedDirModelInterface itself needs a real built rope binary/library + bindings, so it isn't
unit-tested here — just the path-resolution helpers it relies on."""

from __future__ import annotations

import pytest

from rope_dev_tools.validation.model_interfaces import (
    BUILD_DIR_ENV,
    PACKAGE_ROOT_ENV,
    RopePackageNotFoundError,
    _discover_package_root,
    _env_build_dir,
    _resolve_binary_paths,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_resolve_binary_paths_build_dir_override(tmp_path):
    build_dir = tmp_path / "custom-build"
    _touch(build_dir / "rope")
    _touch(build_dir / "librope.so")

    exe, lib = _resolve_binary_paths(tmp_path / "unrelated-package-root", build_dir=build_dir)
    assert exe == build_dir / "rope"
    assert lib == build_dir / "librope.so"


def test_resolve_binary_paths_build_dir_with_bin_lib_subdirs(tmp_path):
    """A packaged/installed release layout (e.g. an extracted rope_framework-*-linux-x86_64-cpu
    tarball) has bin/+lib/ subdirectories rather than the binary/library sitting flat in build_dir."""
    build_dir = tmp_path / "rope_framework-0.4.0-linux-x86_64-cpu"
    _touch(build_dir / "bin" / "rope")
    _touch(build_dir / "lib" / "librope.so")

    exe, lib = _resolve_binary_paths(tmp_path / "unrelated-package-root", build_dir=build_dir)
    assert exe == build_dir / "bin" / "rope"
    assert lib == build_dir / "lib" / "librope.so"


def test_resolve_binary_paths_build_dir_missing_binary_raises(tmp_path):
    build_dir = tmp_path / "custom-build"
    build_dir.mkdir()
    with pytest.raises(RopePackageNotFoundError, match=str(build_dir)):
        _resolve_binary_paths(tmp_path / "unrelated-package-root", build_dir=build_dir)


def test_resolve_binary_paths_falls_back_to_package_root_build_layout(tmp_path):
    root = tmp_path / "rope-framework"
    _touch(root / "build" / "rope")
    _touch(root / "build" / "librope.so")

    exe, lib = _resolve_binary_paths(root)
    assert exe == root / "build" / "rope"
    assert lib == root / "build" / "librope.so"


def test_resolve_binary_paths_falls_back_to_package_root_lib_bin_layout(tmp_path):
    root = tmp_path / "rope-framework"
    _touch(root / "bin" / "rope")
    _touch(root / "lib" / "librope.dylib")

    exe, lib = _resolve_binary_paths(root)
    assert exe == root / "bin" / "rope"
    assert lib == root / "lib" / "librope.dylib"


def test_resolve_binary_paths_no_layout_matches_raises(tmp_path):
    root = tmp_path / "rope-framework"
    root.mkdir()
    with pytest.raises(RopePackageNotFoundError, match=str(root)):
        _resolve_binary_paths(root)


def test_discover_package_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(PACKAGE_ROOT_ENV, str(tmp_path))
    assert _discover_package_root() == tmp_path


def test_discover_package_root_requires_binary_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(PACKAGE_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "rope-framework"
    _touch(root / "python" / "rope.py")
    # no build/rope or bin/rope -- default require_binary=True should fail to discover it
    with pytest.raises(RopePackageNotFoundError):
        _discover_package_root()


def test_discover_package_root_require_binary_false_skips_binary_check(tmp_path, monkeypatch):
    monkeypatch.delenv(PACKAGE_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "rope-framework"
    _touch(root / "python" / "rope.py")

    assert _discover_package_root(require_binary=False) == root


def test_env_build_dir_reads_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(BUILD_DIR_ENV, str(tmp_path))
    assert _env_build_dir() == tmp_path


def test_env_build_dir_none_when_unset(monkeypatch):
    monkeypatch.delenv(BUILD_DIR_ENV, raising=False)
    assert _env_build_dir() is None
