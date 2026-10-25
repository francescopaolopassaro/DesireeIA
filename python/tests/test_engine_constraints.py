"""JSON schema constraint, scoring and logit bias (native engine 0.3.0).
Needs the native library and a model (DESIREEIA_TEST_MODEL_PATH); skipped otherwise."""

import json
import math
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
    _lib() is None or not hasattr(_lib(), "desireeia_set_json_schema") or not os.path.isfile(_MODEL),
    reason="needs a native engine with the JSON schema constraint and DESIREEIA_TEST_MODEL_PATH",
)


@pytest.fixture(scope="module")
def model():
    m, _cfg = LocalModel.load_auto(_MODEL)
    yield m
    m.close()


def _generate(model, user, limit=200):
    ids = model.tokenize(model.apply_chat_template([("user", user)], True), add_bos=True)
    model.reset_session()
    tok = model.predict(ids)
    out = []
    for _ in range(limit):
        if model.is_end_of_generation(tok):
            break
        out.append(tok)
        tok = model.next_token()
    return out


def _text(model, ids):
    return "".join(model.token_piece(t) or "" for t in ids)


SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "population": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "kind": {"enum": ["village", "town", "city"]},
    },
    "required": ["city", "population", "kind"],
}


def test_output_always_matches_the_schema(model):
    model.set_json_schema(SCHEMA)
    try:
        for prompt in ("Describe Rome.", "Tell me about a tiny village.", "Say hello."):
            text = _text(model, _generate(model, prompt, 160))
            doc = json.loads(text)                       # parses, whatever the model wanted to say
            assert isinstance(doc["city"], str)
            assert isinstance(doc["population"], int) and not isinstance(doc["population"], bool)
            assert doc["kind"] in ("village", "town", "city")
            assert set(doc) <= {"city", "population", "tags", "kind"}
            assert len(doc.get("tags", [])) <= 3
    finally:
        model.set_json_schema(None)


def test_any_json_mode_and_switching_off(model):
    model.set_json_schema("json")
    try:
        json.loads(_text(model, _generate(model, "Give me a JSON with two fields.", 120)))
    finally:
        model.set_json_schema(None)
    free = _text(model, _generate(model, "Say hi.", 20))
    assert free                                           # free text again once it is off


def test_bad_schema_is_refused_with_a_message(model):
    with pytest.raises(ValueError):
        model.set_json_schema({"type": "bogus"})
    with pytest.raises(ValueError):
        model.set_json_schema("{not json")


def test_score_tokens_matches_the_distribution(model):
    ids = model.tokenize("The quick brown fox jumps over the lazy dog.", add_bos=True)
    lp, en = model.score_tokens(ids, 1)
    assert len(lp) == len(en) == len(ids) - 1
    assert all(v <= 1e-4 for v in lp)                     # log-probabilities
    assert all(v >= -1e-4 for v in en)                    # entropies
    # a sentence the model knows is less surprising than the same words shuffled
    shuffled = model.tokenize("dog the over lazy jumps fox brown quick The.", add_bos=True)
    lp2, _ = model.score_tokens(shuffled, 1)
    assert sum(lp) > sum(lp2)
    # scoring twice gives the same numbers (the cache is reused or rebuilt, never mixed)
    lp3, _ = model.score_tokens(ids, 1)
    assert all(abs(a - b) < 1e-3 for a, b in zip(lp, lp3))


def test_score_text_continuation_and_ranking(model):
    tokens, lp, en = model.score_text(" Paris", "The capital of France is")
    assert tokens and len(lp) == len(tokens) == len(en)
    _, lp_wrong, _ = model.score_text(" Tokyo", "The capital of France is")
    assert sum(lp) > sum(lp_wrong)


def _first_token(model, user):
    ids = model.tokenize(model.apply_chat_template([("user", user)], True), add_bos=True)
    model.reset_session()
    return model.predict(ids)


def test_logit_bias_forbids_and_forces(model):
    first = _first_token(model, "Say hi.")
    model.set_logit_bias({first: -100.0})
    try:
        assert _first_token(model, "Say hi.") != first
    finally:
        model.set_logit_bias(None)
    assert _first_token(model, "Say hi.") == first       # off again: the original draw
    other = (first + 17) % 1000
    model.set_logit_bias({other: 100.0})
    try:
        assert _first_token(model, "Say hi.") == other
    finally:
        model.set_logit_bias(None)
