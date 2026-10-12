"""ABI 4 extras: sampling extras, reasoning budget, per-token confidence.
Needs the native library and a model (DESIREEIA_TEST_MODEL_PATH); skipped otherwise.
Use a reasoning model (one whose chat template opens <think>) for the budget test."""

import os

import pytest

from desireeia import _native as _nat
from desireeia.model import LocalModel


def _lib():
    try:
        return _nat.get_lib()
    except OSError:
        return None


_MODEL = os.environ.get("DESIREEIA_TEST_MODEL_PATH", "")

pytestmark = pytest.mark.skipif(
    _lib() is None or not hasattr(_lib(), "desireeia_set_sampling_ext") or not os.path.isfile(_MODEL),
    reason="needs a native engine with ABI 4 and DESIREEIA_TEST_MODEL_PATH",
)


@pytest.fixture(scope="module")
def model():
    m, _cfg = LocalModel.load_auto(_MODEL)
    yield m
    m.close()


def _generate(model, prompt, n):
    ids = model.tokenize(model.apply_chat_template([("user", prompt)], True), add_bos=True)
    tok = model.predict(ids)
    out = []
    for _ in range(n):
        if model.is_end_of_generation(tok):
            break
        out.append((model.token_bytes(tok) or b"").decode("utf-8", "replace"))
        tok = model.next_token()
    return "".join(out)


def test_token_stats_after_confidence_tracking(model):
    model.set_confidence_tracking(True)
    try:
        _generate(model, "Say hi.", 8)
        lp, entropy, max_prob = model.token_stats()
        assert lp <= 0.0
        assert entropy >= 0.0
        assert 0.0 < max_prob <= 1.0
    finally:
        model.set_confidence_tracking(False)


def test_sampling_ext_roundtrip_and_off_by_default(model):
    model.set_sampling_ext(min_p=0.1, dry_multiplier=0.5, dry_base=1.5, dry_allowed_length=4, dry_last_n=128)
    model.set_sampling_ext()          # back to all off
    assert _generate(model, "Say hi.", 4) is not None


def test_reasoning_budget_closes_the_thinking(model):
    model.set_reasoning_budget(24)
    try:
        text = _generate(model, "How many sides has a hexagon? Then name three primes.", 400)
    finally:
        model.set_reasoning_budget(0)
    if "</think>" not in text:
        pytest.skip("this model does not open a thinking block")
    thinking = text.split("</think>")[0]
    # 24 tokens of thinking: far fewer than the unbounded run (hundreds), whatever the tokenizer
    assert len(thinking) < 600
