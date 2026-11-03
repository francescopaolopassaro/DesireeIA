# Changelog

## NuGet `DesireeIA` 0.4.0 · PyPI `desireeia` 0.4.0 · `desireeia-server` 0.4.0 · engine 0.4.0 (ABI 6)

Decision models: the engine now runs Laya, a model that answers typed questions with a probability per option instead of writing text. Additive: a separate backend that is only reached when the model path is a decision-model folder; every other model, and every existing call, behaves exactly as in 0.3.1. Wrappers built for 0.3.1 keep working against this engine; the new calls say "needs interface 6" on an older one.

### Added — decision models

- A ModernBERT encoder (rotary positions, gated-GELU feed-forward, bias-free norms, alternating full and sliding-window attention) with half-precision weights widened row by row, and the typed decision head (`choice`, `score`, `noul`; one mask marker per option; per-bucket temperature calibration read from the checkpoint). Loaded straight from the Hugging Face folder: no conversion step.
- A tokenizer driven by `tokenizer.json` (byte-level BPE with the full GPT-2 pre-tokenization pattern, added tokens with their strip flags). Known gaps: no Unicode NFC normalization of the input, and non-ASCII letter/number classification is table-based. The multilingual checkpoint (a different tokenizer) is not supported yet.
- `desireeia_decide` (ABI 6): JSON request in, JSON answers out, two-call buffer convention; a request repeated back to back runs the model once. `NOT_SUPPORTED` when the model is not a decision model, `PARSE` with the reason in `desireeia_last_error` for an invalid request.
- C#: `LocalModel.Decide(state, questions)` with `DecisionQuestion.Choice/Score/NoUl`, `DecisionAnswer`, `DecisionResult`; `DecideJson` for the raw form. Python: `LocalModel.decide(state, questions)`. Server: `POST /v1/decide`; a folder holding a complete decision model in the models directory is listed and served like a checkpoint file.
- Chat on a model that cannot generate (no tokenizer) now answers 400 pointing at `/v1/decide` instead of an opaque 500.

### Verified

- Same weights, same inputs, against the reference implementation (7 cases: all three question types, a structured state, accents/emoji/quotes, a 1024-token input that hits truncation): same token counts, same decisions, largest logit difference 5e-6.
- CPU only, 22 logical cores: 0.13–0.25 s for a 60-token input and 2.6–6.7 s for 1024 tokens (benchmarks in `python/examples/decide_benchmark.py` and `DesireeIA.Bench decide`). The first version took 0.46 s and 18 s: the matrix product now works on token tiles that stay in cache with a 4x2 AVX2 kernel, attention is vectorized, and a decision model uses every logical core unless the plan fixes the thread count (the auto plan is sized for token-by-token decoding and stops at about 8). A GPU path for the encoder is the next step.

### Packaging

- `dotnet pack` now refuses to pack a native library that does not export `desireeia_decide`, naming the file: an engine that `Build\build.ps1` has not refreshed (the Linux library needs Docker) can no longer go into the package silently. `-p:SkipEngineCheck=true` overrides it.

## NuGet `DesireeIA` 0.3.1 · PyPI `desireeia` 0.3.1 · `desireeia-server` 0.3.1 · engine 0.3.1 (ABI 5)

Found by using 0.3.0 from Kodinn's prompt laboratory: two diagnostics that cost far more than they should, one feature that was too slow on big vocabularies, and one speed idea made available. Additive: wrappers built for 0.3.0 keep working; the new calls say "needs a native engine 0.3.1" on an older one.

### Faster — JSON schema output no longer slows generation

- The mask of the tokens that keep the document valid was computed by running the automaton over every token of the vocabulary at every step (262k walks on Gemma 3). The vocabulary is now a byte trie: the automaton advances once per shared prefix and a branch is dropped as soon as the automaton dies on it, so a structural position costs a few thousand steps instead of the vocabulary. The set of allowed tokens is exactly the same (a test compares the two for every state of random documents).
- Free text against schema-constrained, tokens per second: MiniCPM5 2B 65 → 64 (0.3.0: 65 → 51), Spark-X2.5 4B 52 → 52 (53 → 40), Qwen2.5-Coder 3B 48 → 48 (49 → 37), Gemma 3 4B 55 → 55 (55 → 29).

### Faster — reading the attention probe

- `GetAttention` copied a layer's whole probe buffer (16,384 positions a head, about 1 MB) for the few hundred positions in use, and allocated a vector the size of it on every call. A pitched copy now moves only the positions in use, and the host buffer is reused. Reading all 36 layers of MiniCPM5 2B after every token: 26 → 54 tok/s (free generation 62). Reading one layer costs about 0.2 ms instead of 1.2 ms.

### Added — speculative decoding from the text itself (off by default)

- `desireeia_set_speculation(ctx, max_draft)` / `desireeia_get_speculation_info`; `SetSpeculation` / `GetSpeculationInfo` (C#), `set_speculation` / `speculation_info` (Python), `--speculation N` (server). With greedy sampling and nothing that must see every token on its own, the last four tokens are looked up earlier in the context (the occurrence nearest to where a copy is expected to continue wins) and what followed them is tried: one batched pass checks the guess, which is kept up to the first position where the model chose differently. The text is the one the model would have written.
- A gate keeps it free where the text does not repeat: it closes after a few rounds that gained less than 0.7 tokens, or after 24 tokens without a guess, and the pipelined GPU path takes over again.
- Measured on an RTX 1000 Ada: a check pass costs about 3.5 one-token steps and 30–50% of the guesses are kept, so the gain is modest (Spark-X2.5 4B copying a file 49 → 53–59 tok/s, MiniCPM5 2B none, MiniCPM5 2B on the CPU +5%). Outputs identical to speculation off on every tested text. Off by default; documented as aimed at long verbatim copies and CPU-only machines.

## NuGet `DesireeIA` 0.3.0 · PyPI `desireeia` 0.3.0 · `desireeia-server` 0.3.0 · engine 0.3.0 (ABI 5)

What the model considered besides what it wrote, and where it looked when it wrote it.
Both are diagnostics: **off by default, additive, and with them off the engine is not
slower** — measured below. Wrappers built for ABI 3 or 4 keep working against this engine;
the new calls say "needs ABI 5" on an older one.

### Added — output valid against a JSON schema

- `desireeia_set_json_schema(ctx, schema_json)`: the whole output becomes a document valid against the schema. A byte-level automaton follows the generated text (one stack per live alternative, so `anyOf` / `oneOf` / type lists need no backtracking); a token is allowed when its decoded piece keeps at least one stack alive, and the end of generation only when the document is complete. The answer always parses and always has the required keys and types.
- Supported: `type` (also as a list), `properties` / `required` / `additionalProperties`, `items` / `minItems` / `maxItems`, `minLength` / `maxLength`, `enum` / `const`, `anyOf` / `oneOf`, a single-entry `allOf`, `$ref` to `#` and `#/$defs/...` / `#/definitions/...` (recursion included), and `"json"` for any JSON value. Accepted but not enforced: `pattern`, `format`, `minimum` / `maximum`, `uniqueItems`, `prefixItems`. An object that lists `properties` takes only those keys unless `additionalProperties` allows more (at most 64 properties per object).
- Masks are cached by automaton state, and a free-text string takes the plain-piece fast path. Measured on four models: every answer parsed with all the required keys; tokens per second fall by about a quarter while a schema is set (the draw is on the host), Gemma 3 4B with its 262k-token vocabulary more.
- `SetJsonSchema` (C#), `set_json_schema` (Python). `desireeia-server`: `response_format` (`json_object`, `json_schema`) is enforced by the engine; a schema it cannot enforce is a 400 on `response_format`.

### Added — scoring text the model did not write

- `desireeia_score_tokens(ctx, tokens, n, from_index, out_logprob, out_entropy)`: the log-probability of `tokens[from_index..]` given what precedes each, and the entropy (nats) of the distribution at each position. Nothing is generated; the cache ends up holding the text so a following predict that starts with it reuses it. Dense (attention) models.
- `ScoreTokens` (C#), `score_tokens` / `score_text` (Python); `desireeia-server`: `POST /v1/score` with `text` (tokens, log-probabilities, entropy, total, mean, perplexity) or `candidates` (ranked by likelihood).

### Added — logit bias and several answers

- `desireeia_set_logit_bias(ctx, ids, bias, n)`: added to those tokens' logits before every draw; `-100` forbids a token. `SetLogitBias` (C#), `set_logit_bias` (Python), the OpenAI `logit_bias` field in the server.
- `desireeia-server`: `n` up to 8 on `/v1/chat/completions` (not streamed): the prompt is prefilled once, the answers are drawn one after the other, the usage counts the prompt once.

### Faster — prompt processing on the CPU

- Weights are unpacked and interleaved with vector instructions (an 8×8 transpose of 32-bit words, nibble and bit-plane decode of Q4_0 / Q4_K / Q5_K / Q6_K): the per-call unpack costs a third to a half of what it did, which is most of a short prompt. Bit-identical results.
- The chain of dependent dot-product instructions in the Q4_K / Q5_K / Q8_0 kernel is split in two (the ports were idle waiting for the previous result), the shift correction is the accumulators' starting value, tokens are taken in tiles whose activations stay in the L2, and the q / k / v and gate / up products quantize their shared input once. Same integers, same tokens, same probabilities on four models; a 16-row kernel was tried and dropped (slower with this compiler's register allocation).

### Added — top candidates per token

- `desireeia_set_top_candidates(ctx, k)` / `desireeia_get_top_candidates(...)`: with
  confidence tracking on, the `k` (0–16) most likely tokens of the distribution each token was
  drawn from, best first, with their log-probability over the whole vocabulary. The first is the
  chosen token at temperature 0 and its probability equals `max_prob`.
- `SetTopCandidates` / `GetTopCandidates` (C#), `set_top_candidates` / `top_candidates` (Python).
- Costs one comparison per vocabulary entry per token **only while on**; nothing while off.

### Added — attention probe

- `desireeia_set_attention_probe`, `desireeia_get_attention_shape`, `desireeia_get_attention`:
  while on, the attention weights of the last **decoded** token are kept for every layer and head
  and read back after each token (per head, or `head = -1` for the mean over the heads; they sum to 1;
  outside a sliding window they are 0). Up to 16384 positions are recorded.
- Works on the GPU (quantized KV cache, the default — one small extra kernel per layer, also inside the
  captured whole-layer graphs) and on the CPU. A float KV cache on the GPU and MLA models are not covered
  (the layer reports nothing). A prefill is not captured.
- `SetAttentionProbe` / `GetAttentionShape` / `GetAttention` (C#), `set_attention_probe` /
  `attention_shape` / `attention` (Python).
- Turning it on keeps the token draw on the host (no draw ahead of the probe). Output is identical with the
  probe on or off.

### Faster — decode attention for grouped-query models

- The GPU decode attention gave every query head its own block, so a model with several query heads per KV head read and dequantized the same
  keys and values once per head. Profiled at a 7.2k-token context (MiniCPM5 2B, 8 query heads per KV head) it was compute-bound: 72% of the
  SM's issue slots, 4% of the memory bandwidth, 490 us per layer. The new kernel handles a whole KV head per block — queries in shared memory,
  one key per thread for the scores, one dimension per thread for the values, online softmax per tile — and writes the same partials the
  existing merge kernel reads.
- Decode tokens per second at 7.2–7.6k tokens of context: MiniCPM5 2B **29 → 50 (+69%)**, Qwen2.5-Coder 3B 27 → 41 (+49%), Spark-X2.5 4B 41 → 47 (+14%),
  Gemma 3 4B 50 → 52 (+4%); at 1.8–1.9k tokens +23%, +15%, +5%, +3%. (One cold-start reading of the new binary, 55.9 tok/s on a short MiniCPM prompt, did not reproduce in twelve further readings: 66.2 against 66.1.) Short prompts are unchanged: the new kernel is used from 96 tokens of context
  on (the graphs are recaptured once, when a context grows past that line), the old one below.
- A different arithmetic order, so a long greedy generation can diverge from 0.2.1 at a near-tie; both kernels are equally close to the CPU path
  (mean chosen-token log-probability difference 0.0096 and 0.0097). `DESIREEIA_CUDA_ATTN_V1=1` selects the old kernel.

### Added — several models in one process

- Two or more models can now live on the GPU in the same process and be used turn by turn (a Lead and a subagent, a chat model and a coding
  model). They share one set of device buffers; the model that is about to run takes them, its KV rows are uploaded and the previous owner's are
  pulled home when its host copy is behind. A single model pays nothing (a pointer comparison). The CUDA backend now shuts down with the LAST
  model, not with the first one destroyed.
- Checked token for token: two models alternating one token at a time produce exactly what each produces alone, and the first keeps working after the
  second is destroyed (`desireeia_probe_test <model> two`, `DSR_MODEL2`).
- Not covered: running them at the same instant from several threads, and mixing a CPU-only model with a GPU one in the same process (the backend is chosen per process).

### Added — prefix store (several prompts kept warm)

- `desireeia_set_prefix_cache(ctx, budget_bytes, min_tokens)` / `desireeia_get_prefix_cache_info`: the session cache remembers only
  the LAST prompt, so a chat app that serves several conversations, roles or subagents through one model prefilled their shared
  system prompt again at every switch. With a budget, the engine keeps the prompt it is about to overwrite in host memory
  (when at least `min_tokens` positions would be lost) and gives a later prompt the longest prefix it shares with any kept one
  back, instead of a prefill. Least recently used entries are dropped when the budget is exceeded.
- **Off by default** (budget 0): nothing is stored and nothing changes. The restored K/V are exactly what a prefill computed;
  as in any session reuse the last prompt token then runs on its own, so generation can differ from a full prefill only by
  floating-point rounding at a near-tie.
- `SetPrefixCache` / `GetPrefixCacheInfo` (C#), `set_prefix_cache` / `prefix_cache_info` (Python).
- Measured (Spark-X2.5 4B, RTX 1000 Ada 6 GB): coming back to a prompt of 1169 tokens prefilled in 562 ms with the live session
  alone and in 110 ms with the store; at 7629 tokens **3481 ms against 270 ms**. The price is paid when a prompt is
  replaced: about +175 ms to keep a 7.6k-token prompt (1 GB of host memory for two such prompts).
- The store can be written to disk (`desireeia_save_prefix_cache`, `desireeia_load_prefix_cache`; `SavePrefixCache` / `LoadPrefixCache`, `save_prefix_cache` / `load_prefix_cache`): one file per kept prompt, checked against the model and cache layout on load, so a chat app's system prompt survives a restart.
- Saving and restoring a session image (and the store) now moves only the positions in use between the GPU and the host,
  layer by layer, instead of the whole allocation.

### Added — token probabilities in `desireeia-server` (0.3.0)

- `/v1/chat/completions` accepts `logprobs: true` and `top_logprobs` (0–16): every choice carries
  `logprobs.content[]` with `token`, `logprob`, `bytes` and `top_logprobs` — in the response and, one entry per chunk, in the stream.
  The server used to refuse the field. `/v1/completions` (the legacy shape) still refuses it, with a message pointing to chat completions.
- Needs an engine with ABI 5; on an older one the request fails with a clear "not supported" error.
- Probabilities are tracked only for requests that ask for them (the setting is reset for the next request).

### Measured (RTX 1000 Ada 6 GB laptop, old and new engine alternated, 20 s rest between runs)

| Model | Prompt | Old engine | New engine, features off |
|---|---|---|---|
| Spark-X2.5 4B Q4_K_M | 32 tokens | 55.4 / 55.8 tok/s | 56.0 / 55.9 tok/s |
| Spark-X2.5 4B Q4_K_M | 792 tokens (prefill ~400 ms) | 50.3 / 50.1 tok/s | 50.5 / 50.4 tok/s |
| Qwen2.5-Coder 3B Q8_0 | 46 tokens | 50.2 / 50.2 tok/s | 50.1 / 50.1 tok/s |
| Gemma 3 4B Q4_K_M | 24 tokens | 58.4 / 58.6 tok/s | 58.3 / 58.2 tok/s |
| MiniCPM5 2B Q8_0 | 24 / 1824 / 7224 tokens | 67.0 / 50.1 / 29.3 tok/s | 66.9 / 50.0 / 29.2 tok/s |

Longer prompts (Spark-X2.5 4B, same GPU, new engine): prefill stays linear at about 2200 tokens/s — 1932 tokens in 0.9 s,
3832 in 1.7 s, 7632 in 3.5 s — and decode slows from 55 to 49, 46 and 41 tok/s at those contexts.

Features on (Spark 4B): top candidates 54.0 vs 54.1 tok/s with confidence tracking alone; attention probe 52.2 tok/s
against 55.2 (about −5%).

### Checked

- Native self test 82 cases (3 new: top candidates), agentic test 131, session test, on the GPU build and on a CPU-only build;
  server 193 tests (7 new: probabilities); Python 56; C# 44.
- `desireeia_probe_test <model> check`: top candidates sorted and consistent, attention rows normalised on every
  sampled layer and head, generated tokens identical with the features on and off — Spark 4B, Gemma 3 4B (sliding-window layers),
  Qwen2.5-Coder 3B, MiniCPM 2B; GPU and CPU agree on where the model looked.
- The C# test classes that load a model now share one xUnit collection: the CUDA backend
  holds one global scratch/KV cache, so two models cannot run on the GPU in the same process at once).

### Notes

- New benchmarks: `desireeia_probe_test <model> [speed|stats|topk|attn|check|dump]` and `desireeia_speed_test` (the same
  source without the new calls, so it also runs against an older engine). `DSR_REPEAT=N` lengthens the prompt, `DSR_BACKEND=cpu`
  forces the CPU.
- Prefill ideas collected for the next releases are in `docs/performance_plan.md` ("Prefill: nuovi punti").

## NuGet `DesireeIA` 0.2.1 · PyPI `desireeia` 0.2.1 · `desireeia-server` 0.2.1 · engine 0.2.1 (ABI 4)

Found by running small local models as an agent, and what their logs showed:
a reasoning model spending the whole token allowance thinking, sentences
repeated until the turn was cut, and no way to tell a guessed word from a
known one. Everything is off by default and additive: wrappers built for
ABI 3 keep working against this engine.

### Added — reasoning budget

- `desireeia_set_reasoning_budget(max_tokens, open_tag, close_tag)`: after
  `max_tokens` tokens inside the thinking block (it starts at `open_tag`,
  whether the prompt already ends with it or the model writes it) the engine
  writes `close_tag` itself and the model goes on with its answer. Measured
  on Spark-X2.5-4B, same prompt: 232 tokens unbounded (thinking, then the
  answer) against 110 with a budget of 60, same correct answer.
- `set_reasoning_budget` (Python), `SetReasoningBudget` (C#); the server
  reads `reasoning_budget` from the request.

### Added — min-p and DRY

- `desireeia_set_sampling_ext`: `min_p` (drop candidates below that fraction
  of the best one's probability) and DRY (a token that would extend a
  sequence already written in this generation loses
  `multiplier * base^(run - allowed_length)` from its logit; the prompt is
  not looked at, and it is suspended while a constrained tool call is
  written). A separate struct, so ABI 3 wrappers are not affected.
- DRY also lowers the odds of digits or names that legitimately repeat (a
  run on Spark with multiplier 0.8 wrote "22, 32, 132" for three primes):
  keep the multiplier low and use it against loops, not by default.
- `set_sampling_ext` (Python), `SetSamplingExt` (C#); the server reads
  `min_p` and `dry_multiplier` from the request (`min_p` used to be rejected).

### Added — per-token confidence

- `desireeia_set_confidence_tracking` / `desireeia_get_token_stats`: for the
  last token drawn, its log-probability, the entropy of the whole
  distribution and the probability of the most likely token. A flat
  distribution marks a guess, e.g. an invented file name. Costs one pass over
  the logits per token and keeps the draw on the host.
- `set_confidence_tracking` / `token_stats` (Python),
  `SetConfidenceTracking` / `GetTokenStats` (C#).

### Tests

- Native self-test: 79 cases (new: min-p cuts the tail, DRY breaks a repeated
  run / ignores the prompt / is suspended, entropy and log-probability on a
  uniform and a peaked distribution). Python: the three features on a real
  model. C#: same.

## NuGet `DesireeIA` 0.2.0 · PyPI `desireeia` 0.2.0 · `desireeia-server` 0.2.0 · engine 0.2.0 (ABI 3)

### Added — context memory: big files and tool output stay on SSD, not in the prompt

- A file or tool result pasted into the conversation was paid for on every
  later turn in prompt tokens and in KV cache, which lives in VRAM next to
  the weights. The engine now keeps them in a per-session store on SSD
  (own format: Brotli-compressed 40-line blocks, an append-only term index,
  optional embedding rows) and the model gets a short stub: handle, size,
  outline, first lines. It then reads line windows or runs a hybrid BM25 +
  vector search. Measured: a 1 MB source file that would cost 470k prompt
  tokens is worked on within 5k; a 10 MB file is indexed in under 3 s and
  searched in about 40 ms, RAM use bounded by fixed buffers.
- The file on disk stays the source of truth: edits are written straight
  back to it (line endings kept, long lines never truncated), a change made
  outside is picked up on the next access, and an edit based on line
  numbers the file no longer has is refused. Uploads with the same name
  replace the item, streams append to it.
- Sessions idle longer than the TTL are deleted from disk by the engine.
- API: `desireeia_memory_*` (abi.h); `desireeia.ContextMemory` (Python),
  `DesireeIA.ContextMemory` (C#).
- Server: `/tools/open_file`, `/tools/memory_read`, `/tools/memory_search`,
  `/tools/memory_replace_lines`, `/tools/memory_append`; tool results
  longer than `context_spill_chars` are moved to the store automatically.
  The web UI's `read_file` now opens files this way.

### Added — tool calls constrained at token level

- Once the model writes `<tool_call>`, the sampler may only pick tokens that
  keep the output a valid `{"name": <declared tool>, "arguments": {...}}`,
  then the closing tag, then end of generation. Small models no longer
  drop braces, add stray tags or invent tool names; arguments keep their
  JSON types. Free text is not affected.
- `desireeia_set_tool_constraint`; `set_tool_constraint` (Python),
  `SetToolConstraint` (C#). The server enables it whenever a request
  declares tools.

### Added — the model's own chat template, run natively

- The engine executes the GGUF's Jinja `tokenizer.chat_template` itself
  (Hugging Face settings: trim_blocks, lstrip_blocks), so every model gets
  exactly the prompt format it was trained on, tool roles and tool calls
  included. Byte-identical to jinja2 on 23 real templates (Qwen 2.5/3,
  Llama 3.1/3.2, Gemma 2/3, Mistral, Phi-4, DeepSeek-R1, GLM-4/4.5, Kimi-K2,
  gpt-oss, Granite, Falcon3, OLMo, LFM2, SmolLM3, Spark, MiniCPM). A
  template that raises falls back to the built-in formats.
- `desireeia_apply_chat_template_json` (full OpenAI-shaped messages and
  tool definitions); `apply_chat_template_messages` (Python),
  `ApplyChatTemplateJson` (C#). `apply_chat_template` uses the template too.

### Changed

- Reasoning models now get their native generation prompt (often a
  `<think>` block): leave them enough `max_tokens` (measured: 400 cut Spark
  off before its tool call, 2000 did not).
- Tool-call parsing (server and C#) also recovers truncated JSON and the
  `<arg_key>/<arg_value>` format some models emit natively.
- ABI 3: wrappers 0.2.0 refuse an older engine with a clear message.

## NuGet `DesireeIA` 0.1.4 · PyPI `desireeia` 0.1.4 · engine 0.1.3 (unchanged)

### Fixed — emoji and accented characters broken in streamed output

- Streaming decoded every token on its own. A token does not always hold a
  whole character: byte-fallback tokens (`<0xF0>`, `<0x9F>`, ...) carry one
  byte of a UTF-8 sequence, so an emoji spans up to four tokens and each
  piece became U+FFFD ("Ciao! ���" instead of "Ciao! 😊"). `StreamAsync`
  (C#) and `stream` (Python) now go through a stateful UTF-8 decoder that
  holds incomplete bytes until the character is whole; a sequence cut off
  by the end of generation becomes a single U+FFFD.
- New: `LocalModel.TokenBytes(id)` (C#) and `Model.token_bytes(id)`
  (Python) return a token's raw bytes; `TokenPiece` / `token_piece` are
  unchanged for single-token use.
- The native engine is unchanged: the 0.1.3 binaries ship as they are.

## NuGet `DesireeIA` 0.1.3 · PyPI `desireeia` 0.1.3 · engine 0.1.3

### Fixed — SentencePiece tokenization (Gemma and the other SentencePiece vocabularies)

- Segmentation by pair merges, the way these vocabularies were trained:
  adjacent symbols are merged, the pair whose merged piece scores highest
  first. The engine picked the highest-scoring split of each word instead,
  a different segmentation that shredded words into short pieces ("Hello
  world" came out as "Hello", " w", "or", "ld"; "user" as "us", "er"). The
  model read text in a form it was never trained on, and prompts took about
  twice the tokens: the Kodinn agent system prompt is 7,647 tokens on
  Gemma 3 4B instead of 14,516 - half the prefill time, twice the room in
  the context. `DESIREEIA_SPM_VITERBI=1` restores the old segmentation.
- Control and user-defined tokens (`<start_of_turn>`, `<image>`...) are
  taken whole from the text before segmenting it.
- `tokenizer.ggml.add_space_prefix` is honoured: Gemma declares false, and
  every prompt began with a stray space token before `<start_of_turn>`.

## NuGet `DesireeIA` 0.1.2 · PyPI `desireeia` 0.1.2 · engine 0.1.2

### Added — saved sessions (system prompt without prefill)

A long fixed system prompt was prefilled from scratch every time a process
started. The KV cache of a prefix can now be saved and restored:

| | C ABI | C# (NuGet) | Python (PyPI) |
|---|---|---|---|
| to a file | `desireeia_session_save/load` | `SaveSession` / `LoadSession` | `save_session` / `load_session` |
| in memory (encrypt before disk) | `desireeia_session_save_mem/load_mem` | `SaveSessionBytes` / `LoadSessionBytes` | `save_session_bytes` / `load_session_bytes` |

- The image carries the model identity (file size, GGUF header, shape) and
  the cache layout; anything else is refused and the prompt is just prefilled.
- A restored prefix is reused by the next prompt that starts with the same
  tokens, exactly like the in-process session reuse.
- Same format on every OS and backend: saved with CUDA, restorable on CPU.
- Not available for latent-attention (MLA) and recurrent models.

### Added — control of a running engine (ABI version 2)

| | C ABI | C# | Python |
|---|---|---|---|
| interface version | `desireeia_abi_version` | `DesireeIAEngine.NativeAbiVersion`, checked at load | `abi_version()`, checked at load |
| text of the last error | `desireeia_last_error` | in the exception message | in the exception message, `last_error()` |
| stop a prefill from another thread | `desireeia_cancel` → `DESIREEIA_ERR_CANCELLED` | `Cancel()`, stream `CancellationToken`s, `OperationCanceledException` | `cancel()`, `InterruptedError` |
| prefill progress (layers) | `desireeia_set_progress_callback` | `PrefillProgress` | `set_prefill_progress` |
| logger per model | `desireeia_set_ctx_logger` (and the `desireeia_create` callback) | logger of `Load` | logger of `load` |
| GPUs (name, VRAM total/free, capability) | `desireeia_gpu_count` / `desireeia_probe_gpu` | `DesireeIAEngine.Gpus()` | `gpus()` |
| KV cache sizing | `desireeia_reserve_context` / `desireeia_trim_cache` | `ReserveContext` / `TrimCache` | `reserve_context` / `trim_cache` |

- A wrapper on an older engine now says so instead of failing on a missing
  entry point.
- The logger given to `desireeia_create` belongs to that model; it no longer
  replaces the logger of models loaded before it.

### Changed — prefill on the CPU (every machine without an NVIDIA GPU)

Measured on a 16-core laptop CPU, 512-token prefill:

| model | before | now |
|---|---|---|
| Spark 4B Q4_K_M | 19 tok/s | 89 tok/s |
| Gemma 3 4B Q4_K_M | 38 tok/s | 105 tok/s |
| Qwen2.5-Coder 3B Q8_0 | 20 tok/s | 134 tok/s |

- New int8 matrix kernel for Q8_0/Q4_0/Q4_K/Q5_K/Q6_K: weights unpacked to
  exact integers and interleaved 8 rows at a time, one scale per 32 on the
  activations; the K-quant batch kernels used to process one token at a time.
  Picked at run time: AVX2, AVX-VNNI, AVX512-VNNI (256-bit) on x86, NEON
  dotprod on ARM. `DESIREEIA_CPU_ISA=avx2|none` forces a lower level.
- Prefill attention tiled per KV head (each K/V tile dequantized once for all
  the query heads that share it) with a vectorized exp: no longer slows down
  on long prompts.
- Norms, RoPE, KV writes, activations and residuals of a prefill spread over
  all cores (they were ~20% of the time on one thread).
- Linux/macOS builds use explicit instruction sets (x86-64: AVX2+FMA+F16C;
  Apple Silicon: `-mcpu=apple-m1`; other ARM: armv8 baseline) instead of
  `-march=native`, which made a binary crash on CPUs older than the build
  machine's. `DESIREEIA_CPU_FLAGS` overrides.

### Changed — prefill on the GPU

- Whole-prompt prefill runs layer by layer on the device (tensor cores on
  sm_75+, attention one KV head per block); measured on an RTX laptop GPU
  with an 8360-token agent system prompt: 144 s → 10.7 s time to first token.
- Pipelined FP16 tensor-core GEMM (sm_80+): `cp.async` 3-stage pipeline,
  `ldmatrix` fragments, `mma.sync`, 128x256 tiles for wide matrices. The
  measured FP16 ceiling of the test GPU is 17.8 TMAC/s; the previous WMMA
  kernel reached 31% of it. `DESIREEIA_CUDA_NO_HGEMM=1` restores it.
- Thread blocks ordered so the blocks sharing a slice of weights run back
  to back: each weight is read from VRAM once per product instead of once
  per 128 tokens (the products were memory-bound, not compute-bound).
- **FP16 tensor-core products straight on the quantized weights** (sm_80+),
  the default for Q4_K, Q5_K, Q6_K and Q8_0: a stage copies the stored bytes
  of 64 weights per row and every thread builds its own FP16 fragments in
  registers (two 4/5/6/8-bit values under the exponent of 1024, one
  subtraction, one fma with the block scale and minimum). No per-prefill
  expansion of the layer to FP16 or int8, and no per-block rescale on the
  CUDA cores. Q6_K blocks (210 bytes, not 16-byte aligned) are repacked per
  layer in one pass. Measured against the previous expansion kernels: Q8_0
  identical results, K-quants within 1e-3 relative (FP16 rounding of the
  block scales). `DESIREEIA_CUDA_NO_HKQ=1` (or `_NO_HKQ_Q8`, `_NO_HKQ_Q6`)
  returns to the expansion path.
- Tile width picked per product by the number of whole waves it fills on
  the GPU (128 or 256 weight rows per block). `DESIREEIA_CUDA_NO_HKQ_WIDE=1`
  keeps the narrow one.
- Int8 tensor-core products (`mma.sync` m16n8k32/k16, per-32 scales, the
  quantized weights read directly) remain as the path when the FP16 one is
  off; `DESIREEIA_CUDA_INT8_MMA=0` disables them.
- Norm, bias and residual fused into the products: the RMS norm writes the
  FP16 input of the next products itself (same reduction order, identical
  results), biases and the FFN residual are added in the product epilogue,
  the FFN activation goes straight to FP16. `DESIREEIA_CUDA_NO_NORM_F16=1`,
  `DESIREEIA_CUDA_NO_RESID_EPI=1` disable the first and the last.
- Prefill attention on FP16 K/V rewritten: K/V tiles double-buffered with
  `cp.async`, `ldmatrix` fragments, 8 warps per block for head sizes up to
  128, the query tiles with the most keys scheduled first. 8192-token
  prompt, Qwen2.5-Coder 3B: attention 853 -> 683 ms.
- Head sizes up to 128: a second attention kernel keeps the Q fragments in
  registers for the whole key loop, takes 64 keys per tile and evaluates
  the causal/window masks only on the tiles that cross them (683 -> 591 ms
  on the prompt above). `DESIREEIA_CUDA_NO_FA16Q=1` returns to the first.
- Q4_K and Q8_0 models: up and gate in one product - each block computes
  the same 128 rows of both and writes act(gate) * up, so the two FFN
  intermediates never reach memory (+3-4% on Gemma 3 4B). Q5_K/Q6_K keep
  two products (measured faster there). `DESIREEIA_CUDA_NO_GATED=1`
  disables it.
- Shared memory of the quantized products: activation rows and Q8_0 weight
  rows without padding (16-byte chunks XOR-swizzled instead, still free of
  bank conflicts), one half2 of scales per Q8_0 row and stage. The wide
  Q8_0 tile now keeps three pipeline stages instead of two: +3-6% prefill
  on Q8_0 models, +3% on Q4_K.

- Per-matrix preparation of those products (half2 scales, repacked Q6_K)
  kept on the device after the first prompt while at least 1 GB of VRAM
  stays free, instead of being rebuilt for every layer of every prompt;
  dropped with the weights. `DESIREEIA_CUDA_NO_PREP_CACHE=1`.
- The layer-by-layer prefill no longer waits for the GPU after every layer:
  the host mirror of the new KV rows is copied after the last layer, with
  the prompt's single synchronisation, and models that alternate RoPE bases
  build and send each distinct table once per prompt (it used to be rebuilt
  on the CPU at every change of base).
- Prefill chunk sized from the GPU's SM count (64 tokens per SM, 1024 to
  2048): the products fill whole waves of blocks (1280 on a 20-SM GPU,
  +3-4% on Qwen2.5-Coder 3B). `DESIREEIA_CUDA_PREFILL_CHUNK=n` overrides.
- Prefill attention: the output accumulators are rescaled only when a
  row's running maximum moves (after the first key tiles it rarely does;
  multiplying by exactly 1 changed nothing): -4% attention time on an
  8192-token prompt. When the output projection runs on the stored-block
  FP16 kernel the attention writes its FP16 input directly (a weighted mean
  of FP16 values cannot overflow): no float copy, no conversion pass.
  `DESIREEIA_CUDA_NO_ATTN_F16_OUT=1`. The sandwich-norm FFN residual is one
  launch (norm + add).
- Q, K and V in one launch when they share a stored format: on short
  prompts the K/V projections alone fill a fraction of the GPU (+11% at
  512 tokens on MiniCPM5 2B). `DESIREEIA_CUDA_NO_QKV_GROUP=1`.
- Prefill attention with head size 256 (Gemma 3) on the kernel that keeps
  the Q fragments in registers, with 32-key tiles: identical output, attention
  time on an 8192-token prompt 287 → 191 ms (+3% prefill at 2048-8192
  tokens). `DESIREEIA_CUDA_NO_FA16Q_256=1` returns to the previous kernel.
- The device prefill no longer copies the new KV rows to the host cache
  (0.6 GB on an 8k-token prompt of Gemma 3 4B, a serial 86 ms tail after
  the last layer): the device cache is authoritative and the host copy is
  downloaded only before a path that reads it on the host (host-layer
  fallback; session export and cache growth already downloaded).
  `DESIREEIA_CUDA_KV_MIRROR=1` restores the copy. Only the rows the caller
  uses come back from the device (the last one, all of them when every
  position's logits are asked for), and only those are kept for a restore
  on failure (it was a copy of the whole prompt's activations, 84 MB).
  Embedding rows of long prompts are dequantized on the worker threads, and
  the host layer path's input buffer is allocated only when that path runs.
  The GPU now stays busy for the whole prompt, which on power-capped laptop
  GPUs is also what lets the platform raise the power limit:
  Gemma 3 4B at 8192 tokens 2548 → ~2700 tokens/s.
- Prefill attention on sliding-window layers: the window start is rounded
  down to a whole key tile, so the tile boundaries - and each row's
  summation order - no longer depend on where the chunk starts. A restored
  session followed by a prefill of the rest now gives the same bits as a
  full prefill again on Gemma 3 (it drifted at a near tie after ~27 tokens).
- Opt-in int8 products for Q8_0 weights (`DESIREEIA_CUDA_HKQ_I8=1`): the
  activation quantized to int8 per 32 values, m16n8k32 integer products on
  the same tiles as the FP16 kernel, the integer sum turned into a float
  by seeding the accumulator with a magic constant, so each output costs
  two FMAs per block. +3-5% prefill at 2048-8192 tokens on Qwen2.5-Coder
  3B; off by default because int8 activations change the generated text a
  few dozen tokens in (the FP16 path is kept as the reference). The older
  int8 kernel takes the same constant: +5%.

### Changed — generation (decode) on the GPU

- Pipelined decode: the next token is drawn on the device and the step that
  consumes it is queued before the current token is returned to the caller,
  so the GPU no longer waits for the host between two tokens (it idled
  ~0.25 ms per token: synchronisation, sampling, embedding, launch). The
  draw replicates the host sampler step by step - penalties over a device
  copy of the recent history, order, top-k, temperature, top-p - and uses
  the uniform number the host generator would have produced, so the text is
  the same as before: identical token for token, greedy and sampled (tested
  with temperature, top-k, top-p, repetition penalty and fixed seeds). The
  embedding row of the drawn token is read on the device. Any other call on
  the context (a new prompt, sampling settings, sessions, LoRA...) first
  waits for the step in flight and rewinds it (cache position, generator),
  exactly where step-by-step decoding would have been. Used when a single
  context exists, on the fused device path with a Q8_0 tied embedding;
  `DESIREEIA_NO_PIPELINE=1` turns it off. Qwen2.5-Coder 3B Q8_0 49.0 → 49.7
  tokens/s, MiniCPM5 2B Q8_0 64.2 → 65.6.
- Greedy decoding picks the largest logit with a reduction instead of the
  1024-wide per-block sort of the top-k kernel (same token: largest value,
  lowest id among equals).

- Decode attention split over the keys: one warp per key (coalesced reads
  of the Q8_0 cache, shuffle reduction), an online softmax per warp, the
  key range spread over enough blocks to fill the GPU and merged by a
  second small kernel. It replaces one block per head walking the keys
  one thread each. After Kodinn's 14.7k-token system prompt, Gemma 3 4B
  generates at ~43 tok/s instead of ~17. `DESIREEIA_CUDA_NO_ATTN_SPLIT=1`
  restores the previous kernel.
- Output norm and head of a decoded token run on the device right behind
  the last layer: no download of x, norm on the host and upload any more.
  Same logits bit for bit. `DESIREEIA_CUDA_NO_HEAD_DEVICE=1` disables it.
- Sampling: greedy argmax in two vectorizable passes, top-k with a single
  pass over a k-entry heap instead of a partial sort of the vocabulary.
- Norms of a decoded token (one block on one SM, several per layer) read
  the vector once into the registers of 1024 threads, reduce with warp
  shuffles and quantize straight from the registers: 1.49 -> 0.73 ms per
  token on Gemma 3 4B. The sandwich norms are merged with the residual and
  the next norm, and the Q/K head norms with RoPE (same arithmetic, fewer
  launches). `DESIREEIA_CUDA_NO_DEC_FAST=1`, `DESIREEIA_CUDA_NO_DEC_FUSE=1`.
- Models whose layers alternate windows or RoPE bases (Gemma 3): position,
  window and RoPE table of every layer are sent once per token instead of
  before each layer, which stalled the GPU between layer graphs
  (0.3 ms per token).
- The decode logits come back through a page-locked buffer into a vector
  reused across tokens.
- Q6_K mat-vec reads its 2-byte-aligned blocks with 4-byte loads and a
  funnel shift instead of 16-bit loads: the output head of Gemma 3 4B now
  runs at the memory bandwidth (3.85 -> 3.37 ms per token), same results.
- Q/K/V of a decoded token in one launch also when V has a different
  format (Q4_K + Q6_K layers). `DESIREEIA_CUDA_NO_MIXED_GROUP=1`.
- Q4_K mat-vec: the block header (d, dmin, scales) and the weights of a
  sub-block pair are read with 16-byte loads instead of byte and 4-byte
  loads: +6% generation on Gemma 3 4B (53.6 -> 57 tok/s), same results.
- Sampling on the device: when the sampler needs only the largest logits
  (top-k, or greedy, plus one per token a penalty touches - penalties only
  lower a logit) the output head selects them on the GPU and only those
  come back instead of the whole vocabulary. Same token drawn (checked on
  1500 random cases with ties and penalties). `DESIREEIA_CUDA_NO_GPU_TOPK=1`.
- Generation, short context, tok/s: Gemma 3 4B 46 -> 49-52,
  Qwen2.5-Coder 3B 46.5 -> 49, MiniCPM5 2B 59.5 -> 65.
- Result of this round, same laptop (RTX 1000 Ada 6 GB), tok/s at
  512 / 2048 / 8192 prompt tokens:

  | model | before | now |
  |---|---|---|
  | Qwen2.5-Coder 3B Q8_0 | 2405 / 2456 / 2111 | 2781 / 2964 / 2566 |
  | Gemma 3 4B Q4_K_M | 1599 / 1831 / 1807 | 2235 / 2307 / 2397 |
  | MiniCPM5 2B Q8_0 | 3101 / 3135 / 2502 | 3652 / 3665 / 3161 |

  Kodinn's agent system prompt, first run: Gemma (14,717 tokens)
  9.4 -> 6.6 s, Spark (8,360) 5.8 -> 4.4 s, Qwen2.5 (7,724) 4.1 -> 3.2 s,
  MiniCPM5 (8,224) 3.7 -> 2.8 s.
- Prefill attention reads K/V converted to FP16 once per layer (from the
  Q8_0 cache, so the arithmetic is unchanged) instead of dequantizing every
  tile again in every block. `DESIREEIA_CUDA_NO_FA16_KV=1` disables it.
- Host copies of the weights are released once they are in VRAM (restored
  from the model file only if a CPU fallback needs them): process RAM
  2.9 GB -> 1.05 GB with a 4B model. `DESIREEIA_KEEP_HOST_WEIGHTS=1` keeps them.
- Prefill attention in one pass (sm_80+): `mma.sync` FP16 with the scores,
  the online softmax and the output kept in registers, instead of two passes
  over the keys. 8192-token prefill, attention 3.2 s -> 1.24 s; Kodinn's
  agent system prompt: Spark 10.4 -> 8.5 s, Gemma (14.7k tokens)
  14.2 -> 12.4 s, Qwen2.5 7.4 -> 4.8 s, MiniCPM5 7.5 -> 4.4 s.
  `DESIREEIA_CUDA_NO_FA_ATTN=1` falls back to the two-pass kernel.
- KV cache compressed (Q8_0) by default; `DESIREEIA_KV_QUANT=0` restores
  float32.
- CUDA binaries built for sm_75/80/86/89 plus PTX for newer GPUs (was: the
  build machine's GPU only).
- Every ABI function reports native exceptions as error codes
  (`DESIREEIA_ERR_NO_MEM` for allocation failures) instead of crashing the
  host.

### Fixed

- **String metadata was never read.** The GGUF reader kept only the
  architecture and tokenizer names; every other string key was dropped, so
  every lookup of one failed. Consequences, all fixed:
  - `rope.scaling.type` → YaRN never enabled. Models declaring it (e.g.
    Qwen2.5 with a 4× context extension) ran with a plain linear 1/4 RoPE
    scale on every frequency and wrote on-topic but broken text; they now
    answer correctly.
  - `tokenizer.chat_template` → the model's own chat template was never
    detected; the format was guessed from the architecture.
  - `adapter.type` (LoRA check) and the vision projector keys.
- YaRN was also applied only to latent-attention models: it now applies to
  every architecture that declares it, on every RoPE path (CPU, CUDA decode,
  CUDA prefill).
- **RoPE convention of the classic dense GQA family.** Every architecture was
  rotated as split halves; the GGUF files of that family
  (the legacy dense tag, `mistral`, `internlm2`, `baichuan`, `xverse`, `smollm3`,
  `nanbeige`, `arcee`, `cohere2`) are laid out for adjacent pairs, so all of
  them produced degraded text (found with MiniCPM5). Their Q/K rows are now
  reordered at load time, which makes the two conventions coincide on every
  path at no run-time cost.
- The RAM reserve for the KV cache assumed float32 (3.8x the compressed
  cache), pushing weights off RAM for memory never used.

### Build

- `Build/build.ps1` also builds the native self-test and kernel benchmark,
  rebuilds every .NET consumer, runs the native/.NET/Python tests on the
  engine just built (`-SkipTests` to skip), and checks that every copy of
  the engine in the tree is byte-identical to it (a stale copy fails the
  build). The Linux build runs the native self-test on the GCC binary.

## NuGet `DesireeIA` 0.0.3 · PyPI `desireeia` 0.0.5 · PyPI `desireeia-server` 0.0.5 · engine 0.1.1

### Added — automatic configuration

Loading a model used to need hand-tuning: the native planner already picked
backend, threads and RAM budget, but sampling defaulted to greedy (argmax),
which falls into repetitive loops on long generations, and context and reply
length were left to the caller. The new auto-configuration picks everything
for the model on the current machine:

| | C# (NuGet) | Python (PyPI) |
|---|---|---|
| full configuration | `AutoConfigurator.Configure(path)` | `auto_configure(path)` |
| pure decision (testable) | `AutoConfigurator.Compute(hw, plan, traits)` | `compute_configuration(hw, plan, traits)` |
| model facts from the GGUF header | `ModelTraits.Read(path)` | `ModelTraits.read(path)` |
| load + apply | `LocalModel.LoadAuto(path, out config)` | `LocalModel.load_auto(path)` → `(model, config)` |

- **Hardware**: backend (CUDA / Intel / Metal / CPU), threads and RAM budget
  from the native planner, as before.
- **Sampling**: temperature 0.7, top-k 40, top-p 0.9 — never greedy. The
  `general.sampling.*` values suggested by the GGUF file are reported in the
  notes.
- **Context**: target 16384 tokens, never beyond the trained length nor beyond
  what the KV cache fits in memory (conservative upper-bound estimate: 16-bit
  keys/values on every layer; weights counted against RAM because VRAM size
  isn't probed).
- **Reply length**: 2048 tokens, at most a quarter of the context.
- `AutoConfiguration.Notes` explains every choice in plain words.

No native change: the engine binaries are the same as 0.0.2 / 0.0.4.

### Changed

- CLI: without `--temp` generation uses the automatic sampling instead of
  greedy; `--greedy` restores argmax decoding. New `autoconfig <model.gguf>`
  command; `chat` defaults its reply length to the automatic one.
- `desireeia-server`: defaults `temperature 0.7`, `top_p 0.9`,
  `n_predict 2048` (were 0.0 / 0.95 / 512 — greedy by default).
  `--temperature 0` still selects greedy. A `config.json` saved by an earlier
  version keeps the values it stored.

### Fixed

- `desireeia-server`: `CodegenParams.sampling_tuple()` replaced an explicit
  `temperature: 0` with the default (`x or default`); now only a missing value
  is defaulted.

### Tests

- .NET `AutoConfigTests` (8), Python `test_autoconfig.py` (7): defaults, trained
  and memory bounds, alignment, KV arithmetic, real GGUF header read.
- .NET 32/32, Python 41/41, server tests all green.

## NuGet `DesireeIA` 0.0.2 · PyPI `desireeia` 0.0.4 · PyPI `desireeia-server` 0.0.4 · engine 0.1.1

### Added — conversation session (KV prefix reuse)

Until now every `desireeia_predict` call reset the KV cache and prefilled the
whole prompt from scratch. A chat that re-sends its full history each turn
(the normal chat-API pattern) therefore re-processed the entire conversation
at every message, getting slower as the conversation grew.

The engine now keeps the session: `desireeia_predict` compares the new prompt
with the tokens already in the KV cache, keeps the K/V of the shared prefix
and prefills **only the new tokens**.

- **Exact by default.** Only positions computed by a previous *prefill* are
  reused, so the output is identical to a full prefill (verified token by
  token with greedy sampling on gemma-3-4b). In a chat this is still the
  whole history except the last answer.
- **Optional wider reuse** (mode 2) also reuses the tokens generated by
  decode. Faster, but not bit-identical: the single-token decode path and
  the batched prefill path round floating point differently, and at a near
  tie greedy sampling can pick a different word. Measured, documented, off
  by default.
- Reuse is dropped automatically whenever the cached K/V would no longer
  match the token ids: image prefill (vision), loading or clearing a LoRA
  adapter, a failed step. Recurrent (SSM/Mamba) models always prefill in
  full, since their state cannot be rewound to a position.
- Measured on gemma-3-4b Q4_K_M, CPU: second turn of a short chat, prefill
  2.6 s → 0.7 s when the answer is reused too (mode 2); in exact mode the
  saving grows with the length of the history.

New API (all bindings):

| C ABI | C# (`LocalModel`) | Python (`LocalModel`) |
|---|---|---|
| `desireeia_set_session_reuse(ctx, mode)` | `SessionReuse` (`SessionReuseMode.Off / Exact / IncludeGenerated`) | `set_session_reuse(mode)` |
| `desireeia_session_reset(ctx)` | `ResetSession()` | `reset_session()` |
| `desireeia_last_reused_tokens(ctx)` | `LastReusedTokens` | `last_reused_tokens()` |

Existing code needs no change: `Predict`, `StreamAsync` and `ChatStreamAsync`
(and the OpenAI-compatible server, which uses them) get the session
automatically. `desireeia_set_session_reuse(ctx, 0)` restores the previous
always-full-prefill behavior.

### Fixed

- `desireeia-server`: `__version__` reported `0.0.1` while the package was
  `0.0.3`; now aligned (`0.0.4`), so the FastAPI `version` field is correct.
- `desireeia` (Python): `__version__` reported `0.1.0` while the published
  package was `0.0.3`; now aligned (`0.0.4`).
- `desireeia-server` now requires `desireeia>=0.0.4`, so an upgrade of the
  server brings the engine with the session.

### Tests

- New `desireeia_session_test` (native, real model): exact reuse identical to
  a full prefill, prefill-only prefix, mode 2, invalid mode rejected, reuse off.
- New .NET test `Session_ReusesPrefix_AndMatchesFullPrefill` and Python test
  `test_session_reuses_prefix_and_matches_full_prefill` (real model, via
  `DESIREEIA_TEST_MODEL_PATH`).
- Native selftest 74/74, .NET 24/24, Python 34/34.

## NuGet `DesireeIA` 0.0.1 · PyPI `desireeia` / `desireeia-server` 0.0.3

First public releases. See `docs/STATOULTIMOLAVORO16092026.md`.
