"""Scoring endpoint: how likely the model finds a text, token by token, without generating anything.

POST /v1/score
  {"model": optional, "prompt": optional prefix, "text": "..."}
      -> tokens, logprobs, entropies, total / mean log-probability and perplexity
  {"model": optional, "prompt": "...", "candidates": ["yes", "no", ...]}
      -> the same per candidate, and the index of the most likely one (a classifier that never writes free text)

The first token of a text with no prompt has nothing before it to be predicted from, so its log-probability is null.
"""

from __future__ import annotations

import math
from typing import List, Optional

from fastapi import APIRouter, Body, Request

from .errors import InvalidRequestError, NotSupportedError

router = APIRouter(tags=["scoring"])

_MAX_CANDIDATES = 64


def _summary(model, ids: List[int], lps: List[float], ens: List[float], with_pieces: bool) -> dict:
    scored = [v for v in lps if v is not None and math.isfinite(v)]
    total = sum(scored)
    mean = total / len(scored) if scored else None
    out = {
        "tokens": ids,
        "logprobs": [None if (v is None or not math.isfinite(v)) else v for v in lps],
        "entropy": [None if (v is None or not math.isfinite(v)) else v for v in ens],
        "total_logprob": total if scored else None,
        "mean_logprob": mean,
        "perplexity": math.exp(-mean) if mean is not None else None,
    }
    if with_pieces:
        out["pieces"] = [model.token_piece(t) for t in ids]
    return out


@router.post("/v1/score")
def score(request: Request, body: dict = Body(...)):
    prompt = body.get("prompt", "")
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        raise InvalidRequestError("field 'prompt' must be a string", param="prompt")
    with_pieces = body.get("with_pieces", False)
    if not isinstance(with_pieces, bool):
        raise InvalidRequestError("field 'with_pieces' must be a boolean", param="with_pieces")
    text = body.get("text")
    candidates: Optional[list] = body.get("candidates")
    if (text is None) == (candidates is None):
        raise InvalidRequestError("give exactly one of 'text' (score it) or 'candidates' (rank them)")
    if text is not None and not isinstance(text, str):
        raise InvalidRequestError("field 'text' must be a string", param="text")
    if candidates is not None:
        if (not isinstance(candidates, list) or not candidates or len(candidates) > _MAX_CANDIDATES
                or not all(isinstance(c, str) and c for c in candidates)):
            raise InvalidRequestError(f"field 'candidates' must be 1 to {_MAX_CANDIDATES} non-empty strings",
                                      param="candidates")
        if not prompt:
            raise InvalidRequestError("ranking candidates needs a 'prompt' they continue", param="prompt")
    texts = [text] if text is not None else list(candidates)
    if text is not None and not text:
        raise InvalidRequestError("field 'text' must not be empty", param="text")

    slot = request.app.state.slots.resolve(body.get("model"))
    model = slot.ensure_loaded()
    if not hasattr(model, "score_text"):
        raise NotSupportedError("this engine cannot score text (needs a 0.3.0 native engine)")
    try:
        results = slot.score(prompt, texts)
    except RuntimeError as exc:
        raise NotSupportedError(f"scoring is not available for this model: {exc}")

    if text is not None:
        ids, lps, ens = results[0]
        return {"object": "score", "model": slot.model_id, **_summary(model, ids, lps, ens, with_pieces)}
    items = [_summary(model, ids, lps, ens, with_pieces) for ids, lps, ens in results]
    best = max(range(len(items)), key=lambda i: items[i]["total_logprob"]
               if items[i]["total_logprob"] is not None else float("-inf"))
    return {"object": "score.ranking", "model": slot.model_id, "best": best, "candidates": items}
