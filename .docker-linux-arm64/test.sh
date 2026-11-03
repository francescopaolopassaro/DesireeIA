#!/usr/bin/env bash
# Runs the cross-built linux-arm64 engine on a real (QEMU-emulated) aarch64
# container: the native self-test, then, when a decision-model folder is
# mounted at /model, the Python parity check against the recorded reference.
# Expects: /out/linux-arm64/native and /work (from build.sh), /src (the
# repository), and optionally /model and /parity (cpp.py, cases.json, ref.json).
set -euo pipefail
cd /work
export LD_LIBRARY_PATH=/out/linux-arm64/native
./desireeia_selftest > selftest.log 2>&1 || true
SUMMARY=$(grep -E '^passed=' selftest.log | tail -1)
echo "linux-arm64 selftest: $SUMMARY"
case "$SUMMARY" in *"failed=0"*) ;; *) tail -40 selftest.log; exit 1 ;; esac

if [ -d /model ] && [ -d /parity ]; then
  CASES=${CASES:-cases.json}
  REF=${REF:-ref.json}
  PYTHONPATH=/src/python DESIREEIA_NATIVE_LIB=/out/linux-arm64/native/DesireeIALocaleEngine.so \
    python3 /parity/cpp.py /model "/parity/$CASES" "/parity/$REF"
fi
