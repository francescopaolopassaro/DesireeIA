# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Reference decisions for the parity check: Laya's own model code, on the same weights.

Runs `rl_common.DecisionModel` from the Laya repository (the code its authors publish)
over the cases of a JSON file and records tokens, logits, calibrated probabilities and
the act head's output. cpp.py then compares the engine against this file.

Usage:
    python ref.py <laya model folder> <out.json> <cases.json>
with RL_COMMON_DIR pointing at the folder that holds Laya's rl_common.py (the root of a
full download of the repository) and `pip install transformers torch safetensors`.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.environ["RL_COMMON_DIR"])
import rl_common as rc  # noqa: E402  (Laya's code, not part of this repository)
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

model_dir, out_path, cases_path = sys.argv[1:4]
cfg = rc.load_cfg(os.path.join(model_dir, "rl_agent_config.json"))
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
enc = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(model_dir, "encoder")),
                            attn_implementation="eager")
model = rc.DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)
state = {k: v.float() for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}
result = model.load_state_dict(state, strict=False)
assert not result.missing_keys and not result.unexpected_keys, result
model.eval()

LONG = " ".join(["The quarterly report discusses market volatility, bond yields and equity valuations in detail."] * 90)
out = []
for case in json.load(open(cases_path, encoding="utf-8")):
    state_text = LONG if case["state"] == "__LONG__" else case["state"]
    q = dict(case["q"])
    ids, markers = rc.build_sequence(tok, state_text, q, cfg["max_len"], cfg["head_max_len"])
    k = len(markers)
    qtype = rc.QTYPES[q["t"]]
    with torch.no_grad():
        logits, act = model(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), torch.tensor([markers]),
                            torch.ones(1, k, dtype=torch.bool), torch.tensor([qtype]))
    logits = logits[0]
    temperature = cfg["temperature_by_options"].get(rc.temp_bucket(qtype, k), cfg["temperature"][qtype])
    out.append({"name": case["name"], "ids": ids, "markers": markers, "logits": logits.tolist(),
                "probs": torch.softmax(logits / temperature, -1).tolist(), "act": torch.softmax(act[0], -1).tolist(),
                "T": temperature, "state": state_text})
    print(case["name"], len(ids), "tokens")
json.dump(out, open(out_path, "w", encoding="utf-8"))
