#!/usr/bin/env bash
# Builds the native engine for linux-x64 inside the container and copies
# the resulting .so to /out (bind-mounted from the host). Run via
# build.ps1/build.sh at the repo root, not directly.
set -euo pipefail
SRC=/src/src/DesireeIA.LocaleEngine
BUILD=/tmp/build-linux-x64
rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DDESIREEIA_BUILD_TESTS=ON
cmake --build "$BUILD" -j"$(nproc)" --target DesireeIALocaleEngine desireeia_selftest
# The self-test runs here, on the GCC build: the int8 kernels are picked at
# run time (AVX2 / AVX-VNNI / AVX512-VNNI), so this checks the copies GCC
# compiled with per-function target attributes, not just MSVC's.
(cd "$BUILD" && ./desireeia_selftest > selftest.log 2>&1) || true
SUMMARY=$(grep -E '^passed=' "$BUILD"/selftest.log | tail -1)
echo "linux selftest: $SUMMARY"
case "$SUMMARY" in *"failed=0"*) ;; *) tail -40 "$BUILD"/selftest.log; exit 1 ;; esac
mkdir -p /out/linux-x64/native
cp "$BUILD"/DesireeIALocaleEngine.so /out/linux-x64/native/
