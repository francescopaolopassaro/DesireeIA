# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Runs the cases of a parity file through the engine and compares them with ref.py's output.

Usage:
    python cpp.py <laya model folder> <cases.json> <ref.json> [--tolerance 0.05]
The Python package is taken from the repository (python/) next to this folder; set
DESIREEIA_NATIVE_LIB to test a specific native library. DESIREEIA_DECISION_CPU=1 forces
the CPU path (the default uses the GPU of a CUDA build when there is one).

The logits agree to about 1e-5 on the CPU and 4e-3 on the GPU (half-precision matrix
products); the token counts and the decisions must be identical in both.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from desireeia import ExecutionPlan, LocalModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("cases")
ap.add_argument("ref")
ap.add_argument("--tolerance", type=float, default=0.05, help="largest accepted logit difference")
args = ap.parse_args()

messages = []
model = LocalModel.load(args.model_dir, ExecutionPlan(), logger=messages.append)
print("engine:", [m for m in messages if m.startswith("decision model")] or "CPU")
cases = json.load(open(args.cases, encoding="utf-8"))
ref = json.load(open(args.ref, encoding="utf-8"))
worst, failures = 0.0, 0
for case, expected in zip(cases, ref):
    q = case["q"]
    question = {"type": q["t"], "instruction": q["ins"]}
    if q["t"] in ("choice", "score"):
        question["options"] = q["crit"]
    elif q.get("crit"):
        question.update(q["crit"])
    started = time.time()
    answer = model.decide(expected["state"], [question])["answers"][0]
    seconds = time.time() - started
    diff = max(abs(a - b) for a, b in zip(answer["logits"], expected["logits"]))
    best = max(range(len(expected["probs"])), key=lambda i: expected["probs"][i])
    ok = answer["n_tokens"] == len(expected["ids"]) and answer["index"] == best and diff < args.tolerance
    failures += not ok
    worst = max(worst, diff)
    print("%-24s tokens %4d/%4d  max|dlogit| %.6f  decision %s  %.3fs  %s" % (
        case["name"], answer["n_tokens"], len(expected["ids"]), diff, "same" if answer["index"] == best else "DIFFERENT",
        seconds, "OK" if ok else "FAIL"))
print("largest logit difference %.2e, %d failure(s)" % (worst, failures))
model.close()
sys.exit(1 if failures else 0)
