#!/usr/bin/env bash
# Builds the native engine for linux-x64 inside the container and copies
# the resulting .so to /out (bind-mounted from the host). Run via
# build.ps1/build.sh at the repo root, not directly.
set -euo pipefail
SRC=/src/src/DesireeIA.LocaleEngine
BUILD=/tmp/build-linux-x64
rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)" --target DesireeIALocaleEngine
mkdir -p /out/linux-x64/native
cp "$BUILD"/DesireeIALocaleEngine.so /out/linux-x64/native/
