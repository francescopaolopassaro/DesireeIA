# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Decision models (Laya) — usage example and latency benchmark.

Loads a decision-model folder, asks one typed question of every kind, then
times the same question over states of growing length.

Usage:
    python examples/decide_benchmark.py path/to/laya/typed-decisions
    python examples/decide_benchmark.py path/to/laya/typed-decisions --lengths 64,256,1024 --runs 5 --json out.json

The folder is the one downloaded from Hugging Face as it is
(model.safetensors, tokenizer/, encoder/, rl_agent_config.json).
"""

import argparse
import json
import platform
import statistics
import sys
import time

import desireeia
from desireeia import ExecutionPlan, LocalModel

SENTENCE = ("The quarterly report discusses market volatility, bond yields and "
            "equity valuations in detail. ")

CHOICE = {
    "type": "choice",
    "instruction": "What is the main topic of the text?",
    "options": {"cooking": "recipes and food", "finance": "money and markets",
                "sports": "games and athletes", "travel": "trips and places"},
}


def demo(model: LocalModel) -> None:
    """One question of each type over the same state."""
    email = ("Subject: Server down in production\n\nHi team, the main API has been returning "
             "500 errors for the last 20 minutes and customers are complaining.")
    result = model.decide(email, [
        {"type": "choice", "instruction": "Route this email to the right team.",
         "options": {"billing": "payment and invoice questions",
                     "infra": "servers, outages and deployments",
                     "sales": "new customers and pricing"}},
        {"type": "score", "instruction": "Rate the urgency.",
         "options": ["none", "low", "medium", "high", "critical"]},
        {"type": "noul", "instruction": "Customers are affected."},
    ])
    if result is None:
        sys.exit("This model is not a decision model.")
    for answer in result["answers"]:
        probs = ", ".join(f"{o}={p:.2f}" for o, p in zip(answer["options"], answer["probabilities"]))
        print(f"{answer['type']:7s} -> {answer['choice']:<8s} confidence {answer['confidence']:.2f}   [{probs}]")


_salt = 0


def decide_once(model: LocalModel, reps: int) -> tuple:
    # A different state every call: the engine answers an identical request that
    # repeats back to back from memory, which would time as 0 ms.
    global _salt
    _salt += 1
    state = SENTENCE * reps + f"[{_salt}]"
    t0 = time.perf_counter()
    result = model.decide(state, [CHOICE])
    elapsed = time.perf_counter() - t0
    return elapsed, result["answers"][0]["n_tokens"]


def reps_for(model: LocalModel, target: int) -> int:
    """Repetitions of the sentence that make the whole input about `target` tokens."""
    _, small = decide_once(model, 1)
    _, large = decide_once(model, 11)
    per = max(1.0, (large - small) / 10.0)
    overhead = small - per
    return max(1, round((target - overhead) / per))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--lengths", default="64,128,256,512,1024", help="input sizes in tokens, comma separated")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per size (after one warm-up)")
    ap.add_argument("--json", metavar="FILE", help="also write the results to FILE")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = LocalModel.load(args.model_dir, ExecutionPlan())
    load_s = time.perf_counter() - t0
    print(f"DesireeIA {desireeia.__version__}, native engine {desireeia.version()}, "
          f"loaded {args.model_dir} in {load_s:.2f} s")
    print(f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()}\n")

    demo(model)

    print(f"\n{'input tokens':>13} {'median ms':>10} {'min ms':>8} {'tokens/s':>9}")
    rows = []
    for target in [int(x) for x in args.lengths.split(",")]:
        reps = reps_for(model, target)
        decide_once(model, reps)                                  # warm-up
        times, tokens = [], 0
        for _ in range(args.runs):
            elapsed, tokens = decide_once(model, reps)
            times.append(elapsed)
        med = statistics.median(times)
        rows.append({"input_tokens": tokens, "median_ms": med * 1000, "min_ms": min(times) * 1000,
                     "tokens_per_second": tokens / med, "runs": args.runs})
        print(f"{tokens:>13} {med * 1000:>10.0f} {min(times) * 1000:>8.0f} {tokens / med:>9.0f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"engine": desireeia.version(), "load_seconds": load_s, "results": rows}, f, indent=2)
    model.close()


if __name__ == "__main__":
    main()
