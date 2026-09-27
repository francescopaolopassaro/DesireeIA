#!/bin/bash
# Compila e testa il motore per aarch64 (con e senza dotprod) via QEMU.
set -e
# dotprod is an ARMv8.2 extension: GCC only accepts its intrinsics with the
# 8.2 architecture declared (armv8-a+dotprod fails to inline them).
for FLAGS in "-march=armv8.2-a+dotprod" "-march=armv8-a"; do
  BUILD_DIR="/tmp/build_$(echo "$FLAGS" | tr -d ' +-')"
  cmake -S /src/src/DesireeIA.LocaleEngine -B "$BUILD_DIR" -DDESIREEIA_BUILD_TESTS=ON \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$FLAGS" >/dev/null
  cmake --build "$BUILD_DIR" -j4 2>&1 | grep -iE "error|warning: .*neon|warning: .*narrow" || true
  echo "=== $FLAGS ==="
  "$BUILD_DIR/desireeia_selftest" 2>&1 | grep -iE "FAIL|passed="
done
