"""With the windowed KV cache the output is the one the whole cache gives (greedy, a prompt several windows long, decoding past
the point where the local layers slide). Needs DESIREEIA_TEST_MODEL_PATH (a model with local-attention layers) and
DESIREEIA_NATIVE_LIB; each run is a process of its own because the engine reads DESIREEIA_KV_WINDOW when a model opens."""
import json
import os
import subprocess
import sys

import pytest

MODEL = os.environ.get("DESIREEIA_TEST_MODEL_PATH")
LIB = os.environ.get("DESIREEIA_NATIVE_LIB")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from desireeia.model import LocalModel
from desireeia.autoconfig import auto_configure
from desireeia.types import SamplingOptions
cfg = auto_configure(sys.argv[2], None)
m = LocalModel.load(sys.argv[2], cfg.plan)
m.set_sampling(SamplingOptions(temperature=0.0))
base = "Section %d: engineers measured pressure, temperature and flow in the tunnel and wrote %d, %d and %d in the log. "
toks = m.tokenize("".join(base % (i, i * 7, i * 13, i * 31) for i in range(300)))[:2600]
toks += m.tokenize("\nQuestion: which number came third in section 17? Answer:", add_bos=False)
out = [m.predict(toks)]
for _ in range(int(sys.argv[3]) - 1):
    out.append(m.next_token())
print(json.dumps(out))
m.close()
'''


def _run(window, steps):
    env = dict(os.environ, DESIREEIA_KV_WINDOW=window)
    r = subprocess.run([sys.executable, "-I", "-c", SCRIPT, ROOT, MODEL, str(steps)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not MODEL or not LIB, reason="DESIREEIA_TEST_MODEL_PATH / DESIREEIA_NATIVE_LIB not set")
def test_windowed_cache_gives_the_same_tokens_as_the_whole_cache():
    assert _run("1", 300) == _run("0", 300)


SEQ = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from desireeia.model import LocalModel
from desireeia.autoconfig import auto_configure
from desireeia.types import SamplingOptions
cfg = auto_configure(sys.argv[2], None)
m = LocalModel.load(sys.argv[2], cfg.plan)
m.set_sampling(SamplingOptions(temperature=0.0)); m.set_session_reuse(1)
base = "Section %d: engineers measured pressure, temperature and flow in the tunnel and wrote %d, %d and %d in the log. "
T = m.tokenize("".join(base % (i, i * 7, i * 13, i * 31) for i in range(500)))
def gen(toks, n):
    o = [m.predict(toks)]
    for _ in range(n - 1):
        o.append(m.next_token())
    return o
# a long prompt, then one that shares only its first tokens (the cache is cut back and the local layers start over: the
# captured decode graphs must not keep the offsets of the long one), then a long one again, then the same plus a tail
out = [gen(T[:3500], 40), gen(T[:59] + T[2000:2600], 40), gen(T[:59] + T[1000:6000], 40), gen(T[:59] + T[1000:6000] + T[300:340], 40)]
# a session saved for a prefix SHORTER than the cache: with the windowed cache the local layers may have slid past it, and then
# there is nothing to save (the load must never bring back zero rows)
blob = m.save_session_bytes(1200)
if blob:
    m.reset_session()
    m.load_session_bytes(blob)
out.append(gen(T[:59] + T[1000:1200] + T[400:460], 30))
blob = m.save_session_bytes(0)
if blob:
    m.reset_session()
    m.load_session_bytes(blob)
out.append(gen(T[:59] + T[1000:6000] + T[300:340] + T[500:520], 30))
print(json.dumps(out))
m.close()
'''


@pytest.mark.skipif(not MODEL or not LIB, reason="DESIREEIA_TEST_MODEL_PATH / DESIREEIA_NATIVE_LIB not set")
def test_a_cut_back_cache_decodes_like_the_whole_cache():
    def run(window):
        r = subprocess.run([sys.executable, "-I", "-c", SEQ, ROOT, MODEL], env=dict(os.environ, DESIREEIA_KV_WINDOW=window),
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-800:]
        return json.loads(r.stdout.strip().splitlines()[-1])
    assert run("1") == run("0")
