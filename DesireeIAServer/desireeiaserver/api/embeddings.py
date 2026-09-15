"""Embedding endpoints: /v1/embeddings (OpenAI) and /embedding (native)."""

from __future__ import annotations

import math
from typing import List

from fastapi import APIRouter, Body, Request

from .errors import InvalidRequestError, NotSupportedError

router = APIRouter(prefix="/v1", tags=["embeddings"])

EMBED_MODEL_HINT = "model is not a BERT-style encoder; embeddings are not supported"


def _l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _embed_text(model, text: str) -> List[float]:
    ids = model.tokenize(text, add_bos=True)
    if ids is None:
        raise NotSupportedError(EMBED_MODEL_HINT)
    rows = model.embed(ids)
    if rows is None:
        raise NotSupportedError(EMBED_MODEL_HINT)
    dimension = len(rows[0]) if rows else 0
    pooled = [0.0] * dimension
    for row in rows:
        for index, value in enumerate(row):
            pooled[index] += value
    if rows:
        pooled = [value / len(rows) for value in pooled]
    return _l2_normalize(pooled)


def _input_texts(body: dict) -> List[str]:
    value = body.get("input", body.get("content"))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise InvalidRequestError("field 'input' must be a string or a list of strings", param="input")


@router.post("/embeddings")
def embeddings(request: Request, body: dict = Body(...)):
    model = request.app.state.slots.resolve(body.get("model")).ensure_loaded()
    texts = _input_texts(body)

    vectors = []
    prompt_tokens = 0
    for index, text in enumerate(texts):
        ids = model.tokenize(text, add_bos=True) or []
        prompt_tokens += len(ids)
        vectors.append({"object": "embedding", "index": index, "embedding": _embed_text(model, text)})

    return {
        "object": "list",
        "data": vectors,
        "model": request.app.state.slots.resolve(None).model_id,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }