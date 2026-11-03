# Decision-model parity check

Compares the engine with Laya's own code on the same weights, so a change to the encoder,
the head, the tokenizer or a kernel that alters a decision is caught.

| File | What it is |
|---|---|
| `cases.json`, `cases_ml.json` | questions and states for the English and the multilingual checkpoint (one of each question type, a JSON state, accents, emoji, a `<mask>` inside the text, a 1024-token state that is truncated) |
| `cases_short.json`, `cases_ml_short.json` | the four short cases of each, small enough to run on an emulated ARM processor |
| `ref_short.json`, `ref_ml_short.json` | what Laya's code answers for those short cases (token ids, logits, probabilities) |
| `ref.py` | produces such a reference file for any cases file |
| `cpp.py` | runs the cases through the engine and compares |
| `tok_test.cpp`, `tok_compare.py` | compare the tokenizer alone with the Hugging Face `tokenizers` library over 42 strings in many scripts |

## Run it

Download a checkpoint (see the README, "Guide: get Laya from Hugging Face"), then:

```bash
# the engine against the stored references (short cases)
python tools/DecisionParity/cpp.py laya/typed-decisions tools/DecisionParity/cases_short.json tools/DecisionParity/ref_short.json
python tools/DecisionParity/cpp.py laya/multilingual    tools/DecisionParity/cases_ml_short.json tools/DecisionParity/ref_ml_short.json
```

The token counts and the decisions must be identical; the logits agree to about `1e-5` on the CPU and
`4e-3` on the GPU. `DESIREEIA_DECISION_CPU=1` forces the CPU path; `DESIREEIA_NATIVE_LIB` picks the native
library to test.

To regenerate a reference, or to run the long cases, you need Laya's code and `transformers`:

```bash
pip install transformers torch safetensors
hf download convaiinnovations/laya --local-dir laya-full      # the whole repository, for rl_common.py
RL_COMMON_DIR=laya-full python tools/DecisionParity/ref.py laya/typed-decisions ref.json tools/DecisionParity/cases.json
python tools/DecisionParity/cpp.py laya/typed-decisions tools/DecisionParity/cases.json ref.json
```

`Build\build.ps1` runs the short cases on an emulated ARM64 container when `DESIREEIA_TEST_DECISION_MODEL` points
at a model folder.
