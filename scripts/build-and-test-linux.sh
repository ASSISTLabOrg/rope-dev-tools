#!/usr/bin/env bash
# Build and test the Linux x86_64 CPU variant locally.
# Mirrors the build-linux job in .github/workflows/build.yml.
#
# Usage:
#   ./scripts/build-and-test-linux.sh           # configure + build + test
#   ./scripts/build-and-test-linux.sh --clean   # wipe build dir first
#   ./scripts/build-and-test-linux.sh --no-test # skip tests

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"

CLEAN=0
RUN_TESTS=1

for arg in "$@"; do
    case "$arg" in
        --clean)   CLEAN=1 ;;
        --no-test) RUN_TESTS=0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

if [ "$CLEAN" -eq 1 ]; then
    echo "==> Cleaning $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

echo "==> Configuring"
# CI pins gcc-12 inside its Debian container; locally just use whatever gcc is
# on PATH (the CLAUDE.md default is GCC 15 on this machine).
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DROPE_DOWNLOAD_DEPS=ON \
    -DROPE_HARDWARE=cpu \
    -DROPE_USE_LIBTORCH=ON \
    -DROPE_BUILD_TESTS=ON \
    -DROPE_PACKAGE_SUFFIX=linux-x86_64-cpu

echo "==> Building"
cmake --build build --parallel

if [ "$RUN_TESTS" -eq 1 ]; then
    echo "==> Running C++ tests"
    ctest --test-dir build --output-on-failure

    echo "==> Running Python tests"
    pytest tests/python/ -v
fi

echo "==> Done"
