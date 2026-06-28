#!/usr/bin/env bash
# Build and install the pinned NumPP dependency into SciPP/.deps/numpp so that
# `find_package(NumPP)` resolves it. In production NumPP comes from a
# Conan/vcpkg package; this is the local-development convenience path.
#
# Usage: scripts/bootstrap_numpp.sh [path-to-NumPP-source]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUMPP_SRC="${1:-$ROOT/../NumPP}"
PREFIX="$ROOT/.deps/numpp"
BUILD="$ROOT/.deps/numpp-build"

if [[ ! -f "$NUMPP_SRC/CMakeLists.txt" ]]; then
  echo "NumPP source not found at '$NUMPP_SRC'. Pass the path as arg 1." >&2
  exit 1
fi

cmake -S "$NUMPP_SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DNUMPP_BUILD_TESTS=OFF -DNUMPP_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" --target install -j"$(nproc)"
echo "NumPP installed to $PREFIX"
