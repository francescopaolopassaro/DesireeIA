"""Gemma 4: the file loads, the answer to a plain question holds the right word, and the device path (the whole token on the GPU) and the
host path give the same greedy tokens. Needs DESIREEIA_TEST_GEMMA4 (a Gemma 4 GGUF) and DESIREEIA_NATIVE_LIB; the speed floor only runs
on a CUDA machine. Each run is a process of its own because the engine reads DESIREEIA_G4_HOST when a model opens."""
import json
import os
import subprocess
import sys

import pytest

MODEL = os.environ.get("DESIREEIA_TEST_GEMMA4")
LIB = os.environ.get("DESIREEIA_NATIVE_LIB")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not (MODEL and LIB and os.path.exists(MODEL)), reason="needs DESIREEIA_TEST_GEMMA4 and DESIREEIA_NATIVE_LIB")

SCRIPT = r'''
import json, sys, time
sys.path.insert(0, sys.argv[1])
from desireeia.model import LocalModel
from desireeia.autoconfig import auto_configure
from desireeia.types import SamplingOptions
cfg = auto_configure(sys.argv[2], None)
m = LocalModel.load(sys.argv[2], cfg.plan)
m.set_sampling(SamplingOptions(temperature=0.0))
toks = m.tokenize(sys.argv[3])
out = [m.predict(toks)]
t = time.time()
n = int(sys.argv[4])
for _ in range(n - 1):
    out.append(m.next_token())
dt = time.time() - t
text = "".join(m.token_piece(x) or "" for x in out)
print(json.dumps({"ids": out, "text": text, "tps": (n - 1) / dt, "backend": int(cfg.plan.backend)}))
'''

PROMPT = "<|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n<|turn>model\n"


def run(host, n=12):
    env = dict(os.environ)
    if host:
        env["DESIREEIA_G4_HOST"] = "1"
    else:
        env.pop("DESIREEIA_G4_HOST", None)
    p = subprocess.run([sys.executable, "-c", SCRIPT, ROOT, MODEL, PROMPT, str(n)], capture_output=True, text=True, env=env, timeout=900)
    assert p.returncode == 0, p.stderr[-600:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_answers_paris():
    assert "paris" in run(False)["text"].lower()


def test_device_path_equals_host_path():
    a, b = run(False), run(True)
    assert a["ids"][:6] == b["ids"][:6], (a["text"], b["text"])


def test_device_path_speed_floor():
    r = run(False, 24)
    if r["backend"] != 2 and "cuda" not in str(r["backend"]).lower():
        pytest.skip("no CUDA backend")
    assert r["tps"] > 20, r["tps"]
