"""Decision endpoint: typed answers from a decision model (for example Laya), never free text.

POST /v1/decide
  {"model": optional,
   "state": "free text" or any JSON value,
   "questions": [
     {"type": "choice", "instruction": "...", "options": {"label": "description", ...}},
     {"type": "score",  "instruction": "...", "options": ["level 0 text", "level 1 text", ...]},
     {"type": "noul",   "instruction": "...", "true": "optional", "false": "optional"}],
   "calibrate": true,          optional
   "truncate_left": false}     optional: keep the END of a state that does not fit
      -> {"object": "decision", "model": ..., "answers": [
            {"type", "options", "probabilities", "logits", "index", "choice",
             "confidence", "act_probability", "act_probabilities", "n_tokens"}, ...]}

The model scores each option and returns a probability per option, so there is nothing to parse and nothing to
hallucinate. `confidence` is 1 minus the normalized entropy of the answer (0 = undecided, 1 = certain); prefer it over
`act_probability`, which its authors report as unreliable.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Request

from .errors import InvalidRequestError, NotSupportedError

router = APIRouter(tags=["decisions"])

_MAX_QUESTIONS = 32
_TYPES = ("choice", "score", "noul")


def _check_question(index: int, question) -> None:
    where = f"questions[{index}]"
    if not isinstance(question, dict):
        raise InvalidRequestError(f"{where} must be an object", param="questions")
    qtype = question.get("type", question.get("t"))
    if qtype not in _TYPES:
        raise InvalidRequestError(f"{where}.type must be one of {', '.join(_TYPES)}", param="questions")
    instruction = question.get("instruction", question.get("ins", ""))
    if not isinstance(instruction, str):
        raise InvalidRequestError(f"{where}.instruction must be a string", param="questions")
    options = question.get("options", question.get("crit", question.get("levels")))
    if qtype == "choice" and not isinstance(options, (dict, list)):
        raise InvalidRequestError(f"{where}.options must be an object (label -> description) or a list of labels",
                                  param="questions")
    if qtype == "score" and not isinstance(options, list):
        raise InvalidRequestError(f"{where}.options must be a list with one text per level", param="questions")


@router.post("/v1/decide")
def decide(request: Request, body: dict = Body(...)):
    if "state" not in body or body["state"] is None:
        raise InvalidRequestError("field 'state' is required", param="state")
    questions = body.get("questions")
    if not isinstance(questions, list) or not questions or len(questions) > _MAX_QUESTIONS:
        raise InvalidRequestError(f"field 'questions' must be a list of 1 to {_MAX_QUESTIONS} questions",
                                  param="questions")
    for index, question in enumerate(questions):
        _check_question(index, question)
    calibrate = body.get("calibrate", True)
    truncate_left = body.get("truncate_left", False)
    for name, value in (("calibrate", calibrate), ("truncate_left", truncate_left)):
        if not isinstance(value, bool):
            raise InvalidRequestError(f"field '{name}' must be a boolean", param=name)

    slot = request.app.state.slots.resolve(body.get("model"))
    model = slot.ensure_loaded()
    if not hasattr(model, "decide"):
        raise NotSupportedError("this engine cannot run decision models (needs a native engine with interface 6)")
    try:
        result = slot.decide(body["state"], questions, calibrate=calibrate, truncate_left=truncate_left)
    except ValueError as exc:
        raise InvalidRequestError(str(exc))
    except RuntimeError as exc:
        raise NotSupportedError(f"decisions are not available for this model: {exc}")
    if result is None:
        raise NotSupportedError("model is not a decision model; /v1/decide needs one (for example Laya)")
    return {"object": "decision", "model": slot.model_id, "answers": result["answers"]}
