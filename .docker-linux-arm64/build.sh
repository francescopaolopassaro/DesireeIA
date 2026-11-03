#!/usr/bin/env bash
# Cross-builds the native engine for linux-arm64 (generic armv8-a baseline, no
# dotprod: it has to run on any 64-bit ARM Linux) on the x64 container, at
# native speed, and copies the shared library and the self-test to /out
# (bind-mounted from the host). The result is then RUN on an emulated aarch64
# container by test.sh. /out is Build/runtimes, /work holds the test binary.
# Run via Build/build.ps1, not directly.
set -euo pipefail
SRC=/src/src/DesireeIA.LocaleEngine
BUILD=/tmp/build-linux-arm64
rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DDESIREEIA_BUILD_TESTS=ON \
      -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
cmake --build "$BUILD" -j"$(nproc)" --target DesireeIALocaleEngine desireeia_selftest
mkdir -p /out/linux-arm64/native /work
cp "$BUILD"/DesireeIALocaleEngine.so /out/linux-arm64/native/
cp "$BUILD"/desireeia_selftest /work/
