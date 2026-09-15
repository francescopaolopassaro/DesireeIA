"""Tokenizer parity endpoints: /tokenize, /detokenize, /apply-template."""

from __future__ import annotations

from fastapi import APIRouter, Body, Request

from .errors import InvalidRequestError, NotSupportedError

router = APIRouter(tags=["tokenizer"])

_TOKENIZE_MISSING = ("content", "'content' (string) is required for /tokenize")


def _model_for(request: Request, body: dict):
    return request.app.state.slots.resolve(body.get("model")).ensure_loaded()


@router.post("/tokenize")
def tokenize(request: Request, body: dict = Body(...)):
    if "content" not in body or not isinstance(body["content"], str):
        raise InvalidRequestError("field 'content' must be a string", param="content")
    add_special = body.get("add_special", True)
    if not isinstance(add_special, bool):
        raise InvalidRequestError("field 'add_special' must be a boolean", param="add_special")
    with_pieces = body.get("with_pieces", False)
    if not isinstance(with_pieces, bool):
        raise InvalidRequestError("field 'with_pieces' must be a boolean", param="with_pieces")

    model = _model_for(request, body)
    ids = model.tokenize(body["content"], add_bos=add_special)
    if ids is None:
        raise NotSupportedError("model has no recognized tokenizer")
    result = {"tokens": ids}
    if with_pieces:
        pieces = []
        for token_id in ids:
            piece = model.token_piece(token_id)
            pieces.append(piece if piece is not None else None)
        result["pieces"] = pieces
    return result


@router.post("/detokenize")
def detokenize(request: Request, body: dict = Body(...)):
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in tokens
    ):
        raise InvalidRequestError("field 'tokens' must be an array of integers", param="tokens")

    model = _model_for(request, body)
    pieces = []
    for token_id in tokens:
        piece = model.token_piece(token_id)
        if piece is None:
            raise InvalidRequestError(f"token id {token_id} is not valid for this model", param="tokens")
        pieces.append(piece)
    return {"text": "".join(pieces)}


@router.post("/apply-template")
def apply_template(request: Request, body: dict = Body(...)):
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("field 'messages' must be a non-empty array", param="messages")
    pairs = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise InvalidRequestError(f"messages[{index}] must have 'role' and 'content'")
        pairs.append((str(message["role"]), str(message.get("content") or "")))
    add_assistant = body.get("add_assistant", True)
    if not isinstance(add_assistant, bool):
        raise InvalidRequestError("field 'add_assistant' must be a boolean", param="add_assistant")

    model = _model_for(request, body)
    try:
        prompt = model.apply_chat_template(pairs, add_assistant)
    except Exception as exc:
        raise InvalidRequestError(f"chat template failed: {exc}")
    return {"prompt": prompt}