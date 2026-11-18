"""Compatibility matrix: which model files load, and whether what they generate makes sense.

    python tools/compat_matrix.py MODEL.gguf [MODEL.gguf ...]      (or a folder of .gguf files)

For every file: the architecture tag, whether it loads, the mean log-probability of a plain English sentence (a model that reads its own
file wrongly scores far lower than one that does) and whether a greedy answer to a one-line question holds the expected word. Prints a
Markdown table. Needs DESIREEIA_NATIVE_LIB (or the engine next to the package). A file that crashes the engine takes the process down:
each one runs in a process of its own.
"""
import json
import os
import subprocess
import sys

CHILD = r'''
import json, os, sys
sys.path.insert(0, sys.argv[2])
from desireeia.model import LocalModel
from desireeia.autoconfig import auto_configure, read_gguf_metadata
from desireeia.types import SamplingOptions
path = sys.argv[1]
r = {"file": os.path.basename(path), "arch": "?", "loads": False, "score": None, "answer": "", "ok": False, "error": ""}
try:
    r["arch"] = read_gguf_metadata(path).get("general.architecture", "?")
    cfg = auto_configure(path, None)
    m = LocalModel.load(path, cfg.plan)
    r["loads"] = True
    m.set_sampling(SamplingOptions(temperature=0.0))
    _, lp, _ = m.score_text("The quick brown fox jumps over the lazy dog. The capital of France is Paris, and the capital of Italy is Rome.")
    r["score"] = round(sum(lp) / max(1, len(lp)), 2) if lp else None
    toks = m.tokenize("Question: What is the capital of France? Answer in one word.\nAnswer:")
    out = [m.predict(toks)]
    for _ in range(11):
        out.append(m.next_token())
    r["answer"] = "".join(m.token_piece(t) or "" for t in out).strip()[:40]
    r["ok"] = "paris" in r["answer"].lower()
    m.close()
except Exception as e:
    r["error"] = str(e)[:90]
print(json.dumps(r))
'''


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    files = []
    for a in sys.argv[1:]:
        files += [os.path.join(a, f) for f in sorted(os.listdir(a)) if f.endswith(".gguf")] if os.path.isdir(a) else [a]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
    rows = []
    for f in files:
        p = subprocess.run([sys.executable, "-c", CHILD, f, root], capture_output=True, text=True, timeout=900)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"file": os.path.basename(f), "arch": "?", "loads": False, "score": None, "answer": "", "ok": False,
                         "error": f"process died (exit {p.returncode})"})
    print("| file | architecture | loads | mean logprob | answer | sane |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['file']} | {r['arch']} | {'yes' if r['loads'] else 'NO'} | {r['score']} | {(r['answer'] or r['error']).replace('|', '/').replace(chr(10), ' ')} | {'yes' if r['ok'] else 'NO'} |")


if __name__ == "__main__":
    main()
