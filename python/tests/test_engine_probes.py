"""ABI 5 probes: top candidates per token and the attention probe.
Needs the native library (ABI 5) and a model (DESIREEIA_TEST_MODEL_PATH); skipped otherwise."""

import math
import os

import pytest

from desireeia import _native as _nat
from desireeia.model import LocalModel
from desireeia.types import SamplingOptions


def _lib():
    try:
        return _nat.get_lib()
    except OSError:
        return None


_MODEL = os.environ.get("DESIREEIA_TEST_MODEL_PATH", "")

pytestmark = pytest.mark.skipif(
    _lib() is None or not hasattr(_lib(), "desireeia_set_top_candidates") or not os.path.isfile(_MODEL),
    reason="needs a native engine with ABI 5 and DESIREEIA_TEST_MODEL_PATH",
)


@pytest.fixture(scope="module")
def model():
    m, _cfg = LocalModel.load_auto(_MODEL)
    yield m
    m.close()


def _greedy(model):
    s = model.get_sampling()
    model.set_sampling(type(s)(**{**s.__dict__, "temperature": 0.0}) if hasattr(s, "__dict__") else s)


def _generate(model, n, per_token=None):
    ids = model.tokenize(model.apply_chat_template([("user", "Say hi in one short sentence.")], True), add_bos=True)
    model.reset_session()
    tok = model.predict(ids)
    out = []
    for i in range(n):
        if model.is_end_of_generation(tok):
            break
        out.append(tok)
        if per_token:
            per_token(i, tok)
        tok = model.next_token()
    return out


def test_top_candidates_are_sorted_and_consistent(model):
    model.set_confidence_tracking(True)
    model.set_top_candidates(5)
    seen = []
    try:
        _generate(model, 8, lambda i, tok: seen.append((tok, model.top_candidates(), model.token_stats())))
    finally:
        model.set_top_candidates(0)
        model.set_confidence_tracking(False)
    assert seen
    for _tok, top, stats in seen:
        assert top is not None and len(top) == 5
        lps = [lp for _id, lp in top]
        assert lps == sorted(lps, reverse=True)
        assert sum(math.exp(lp) for lp in lps) <= 1.0 + 1e-3
        assert abs(math.exp(lps[0]) - stats[2]) < 1e-3     # top-1 probability == max_prob


def test_top_candidates_off_by_default(model):
    model.set_confidence_tracking(True)
    try:
        _generate(model, 3, None)
        assert model.top_candidates() is None
    finally:
        model.set_confidence_tracking(False)


def test_attention_probe_weights_sum_to_one_and_leave_output_unchanged(model):
    model.set_sampling(SamplingOptions(temperature=0.0))        # greedy: with and without the probe the tokens must match
    plain = _generate(model, 10)
    assert model.set_attention_probe(True)
    try:
        shapes = []

        def look(i, tok):
            shape = model.attention_shape()
            if shape is None:                                # the first token can come out of a prefill: nothing captured
                assert i == 0
                return
            shapes.append(shape)
            layers, heads, positions, _total = shape
            for layer in range(0, layers, max(1, layers // 4)):
                w = model.attention(layer)                   # mean over the heads
                assert w is not None and len(w) == positions
                assert abs(sum(w) - 1.0) < 2e-2 and min(w) >= 0.0
                one = model.attention(layer, 0)
                assert one is not None and abs(sum(one) - 1.0) < 2e-2

        probed = _generate(model, 10, look)
    finally:
        model.set_attention_probe(False)
    assert shapes
    assert probed == plain                                    # the probe never changes what is generated
    assert model.attention_shape() is None                    # off again: nothing is kept


def _long_prompt(model, topic, reps=25):
    text = (topic + " ") * reps
    return model.tokenize(model.apply_chat_template([("user", text + "Summarise it in one sentence.")], True), add_bos=True)


def _run(model, ids, n=8):
    tok = model.predict(ids)
    out = []
    for _ in range(n):
        if model.is_end_of_generation(tok):
            break
        out.append(tok)
        tok = model.next_token()
    return out


def test_prefix_cache_restores_an_earlier_prompt(model):
    assert model.prefix_cache_info() == (0, 0, 0, 0)               # off by default: nothing held
    a = _long_prompt(model, "Daylight looks blue because the atmosphere scatters short wavelengths.")
    b = _long_prompt(model, "A compiler turns source code into machine instructions and optimises them.")
    assert model.set_prefix_cache(512 * 1024 * 1024, 64)
    model.set_sampling(SamplingOptions(temperature=0.0))            # greedy: the two visits must produce the same tokens
    try:
        model.reset_session()
        first = _run(model, a)
        _run(model, b)
        assert model.last_reused_tokens() < len(a) // 2               # B shares almost nothing with A
        again = _run(model, a)
        assert model.last_reused_tokens() >= len(a) - 4               # A came back from the store
        assert again and first                                      # both visits answer (the last prompt token runs on its own, as in any session reuse:
                                                                    # rounding at a near-tie may change a word, so no token-for-token equality here)
        nbytes, entries, hits, restored = model.prefix_cache_info()
        assert hits >= 1 and entries >= 1 and restored >= len(a) // 2 and nbytes > 0
    finally:
        assert model.set_prefix_cache(0)
    assert model.prefix_cache_info() == (0, 0, 0, 0)               # switched off: everything released


def test_prefix_cache_survives_on_disk(model, tmp_path):
    a = _long_prompt(model, "Daylight looks blue because the atmosphere scatters short wavelengths.")
    b = _long_prompt(model, "A compiler turns source code into machine instructions and optimises them.")
    assert model.set_prefix_cache(512 * 1024 * 1024, 64)
    try:
        model.reset_session()
        _run(model, a)
        _run(model, b)                                             # A is kept when B replaces it
        assert model.save_prefix_cache(tmp_path) >= 1
        assert any(p.suffix == ".dskv" for p in tmp_path.iterdir())
        assert model.set_prefix_cache(0)                           # off: the in-memory store is gone
        assert model.prefix_cache_info() == (0, 0, 0, 0)
        assert model.set_prefix_cache(512 * 1024 * 1024, 64)
        assert model.load_prefix_cache(tmp_path) >= 1              # as after a restart
        model.reset_session()
        _run(model, a)
        assert model.last_reused_tokens() >= len(a) - 4            # A came back from the disk copy
        assert model.load_prefix_cache(tmp_path / "does-not-exist") == 0
    finally:
        model.set_prefix_cache(0)
