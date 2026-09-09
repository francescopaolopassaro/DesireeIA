# DesireeIA

A local large language model inference engine for the .NET ecosystem: a native C++17 core exposed through a stable C ABI, with an idiomatic .NET wrapper on top.
<img src="images/DesireeIA.jpg" />
## What it does

DesireeIA loads quantized language models from disk and runs them entirely on local hardware — no network calls, no external services. It targets practical CPU throughput while auto-adapting its execution plan to the hardware it's running on.

## Background

This project started in 2023, out of a simple need: a local inference engine we actually owned, built primarily for the .NET ecosystem and, from the outset, designed to be usable from other technologies too rather than locked into one runtime. Going straight to C++ for the core was a deliberate choice from day one — it's what gives the engine tight control over memory, the best achievable performance on ordinary hardware, and the broadest possible portability across platforms. That decision built on real prior experience with local AI: by that point we'd already evaluated and run thousands of GGUF models from the Hugging Face ecosystem, so the engine's design started from a working understanding of what actually mattered in practice, not from a blank slate.

The project was set aside for a while as other priorities took over, but it came back into focus with the development of a personal AI assistant — Kodinn IA, originally called Offgrid — which needed to run primarily on local models. Development resumed, running alongside the rest of that assistant's libraries rather than as the main focus, until it reached the point of being split out and developed as its own project in the shape it's in today.

## Supported model formats

- **GGUF**
- **safetensors**, including models with Mixture-of-Experts weights streamed from disk

## Supported architectures

Dense causal transformers (RoPE, grouped-query attention, gated/non-gated feed-forward), with per-family behavior handled through a quirks system rather than one implementation per model:

- Gemma, Gemma 2, Gemma 3
- Qwen, Qwen 2, Qwen 3 (+ Qwen2-MoE)
- Mistral and the broader Llama-shaped family (InternLM2, Exaone, SmolLM3, Nanbeige, Baichuan, XVerse, PLaMo, Refact, PanGu-Embedded)
- SeedOss, StableLM, Orion
- Starcoder2 (+ CodeShell), Nemotron, Arcee
- Olmo2, Exaone4 (post-norm topology)
- Cohere2, Falcon (parallel attention/FFN topology, fused QKV)
- GPT-2, BLOOM, MPT (absolute position embeddings / ALiBi, no RoPE)

Beyond the dense family:

- **DeepSeek2** — Multi-head Latent Attention (compressed KV cache, absorbed projections), Mixture-of-Experts with a shared-expert branch and sigmoid gating, YaRN rope scaling
- **Mamba2** — pure state-space model (selective scan, causal depthwise convolution, constant-size recurrent state instead of a growing KV cache)
- **BERT** — encoder-only, bidirectional attention, post-norm topology, per-token embedding output (no next-token generation)

Mixture-of-Experts models route through a tiered expert store (see below) rather than holding every expert in RAM at once.

## Engine architecture

Three layers:

1. **Native core** (C++17, C ABI) — hardware probe, execution planner, quantized matrix kernels, model forward paths, GGUF/safetensors readers, KV cache, tokenizers (SentencePiece Unigram and byte-level BPE), sampler, chat template formatting.
2. **Hardware adaptation layer** — probes the machine and builds an execution plan (thread count, RAM budget, quantization target, backend) before any model is loaded.
3. **.NET wrapper** — P/Invoke bindings plus idiomatic C# types over the native ABI.

### Startup flow

```
DetectHardware()  ->  BuildPlan(modelPath, overrides)  ->  LocalModel.Load(...)
       |                        |                                |
  native hw probe        execution plan                native context + reader
```

At load time the engine:

1. Detects hardware (physical CPU core count — hyperthread siblings are not
   counted as extra cores, since the AVX2 kernels here are memory-bound and
   two threads sharing a physical core's L1/L2 contend rather than add
   throughput —, AVX/AVX2/AVX512/NEON, CUDA/Intel Arc/Metal/NPU acceleration,
   RAM). NUMA topology is not currently detected.
2. Builds an execution plan automatically unless the caller overrides it (RAM budget, thread count, quantization, format).
3. Applies tiered caching for MoE experts:
   - experts streamed from disk with an LRU cache,
   - a learned hot-set (frequently used experts pinned, persisted alongside the model),
   - one-layer-ahead expert prefetch,
   - batch-union expert fetching (each unique expert read once per batch of positions),
   - optional dual-SSD mirroring.

Every part of the plan can be overridden by the caller.

### Storage tier

Models larger than the RAM budget can be served straight from disk instead of
being loaded whole. The tier is configurable per load:

| Mode | Behaviour |
| --- | --- |
| `Off` | Weights are served from RAM only. |
| `Auto` (default) | The tier engages only when the weights do not fit the budget. |
| `Always` | Weights are always streamed, even when they would fit. |

`Auto` decides from a real budget rather than a flat percentage: a system
reserve is left to the operating system, a runtime reserve covers the KV cache,
activations and per-layer dequantization scratch, and the weights are compared
against what remains. A model that fits stays entirely on the RAM path, and the
tier is not constructed at all — nothing sits on the tensor read path and
throughput is unchanged.

The tier never parses the model file itself; it serves misses through the same
reader the rest of the engine uses, so both paths see identical bytes.

```csharp
var plan = new ExecutionPlan
{
    SsdTier = SsdTierMode.Always,
    SsdTierCacheMb = 2048
};
```

`DESIREEIA_SSD_TIER=off|auto|always` overrides the mode for a single run.

### Weight requantization

Decode is bandwidth-bound: every weight is read once per token and the
arithmetic per byte is small, so time spent tracks bytes moved. Q6_K tensors
can optionally be rewritten as Q4_K while loading, which moves about 31% fewer
bytes for those tensors.

This is off by default. It is a quality trade, not a free win, and one the
model's author already weighed — a Q4_K_M file keeps the output projection at
Q6_K precisely because it is the tensor most sensitive to 4-bit. Enable it with
`DESIREEIA_REQUANT_Q6K=1` and measure on your own model.

The quantizer fits each sub-block by minimizing squared reconstruction error
(alternating nearest-level assignment with the closed-form least-squares
regression of the values on their levels, over several starting ranges) rather
than simply spanning min to max, and applies the same treatment to the shared
6-bit scales. The self-test asserts it beats a plain min/max fit.

### KV cache quantization

The KV cache can optionally be stored as Q8_0 (32-value sub-blocks, one fp16
scale each) instead of float32 — roughly 3.76x smaller. Only applies to the
classic (non-MLA) attention path; an MLA model's own compressed latent cache
is already far smaller than a classic per-head cache, so this doesn't touch it.

Off by default, and not just out of caution: measured at 48 and ~400 tokens
of context on this project's own benchmark, it was 15-24% *slower*, not
faster — at those lengths the float cache already fits comfortably in CPU
cache, so there's no real memory-bandwidth pressure to relieve, and every
read pays a real int8-to-float decode cost with nothing to show for it. The
crossover point where a smaller cache would start winning, if there is one
on typical hardware, needs a much longer context than what's been tested
here. Enable it with `DESIREEIA_KV_QUANT=1` to measure on your own
workload — long-context use is the case worth trying it on.

### Memory locking

The largest weight tensors (the token embedding, and the output head when a
model has a separate one) can optionally be locked resident with the OS —
`VirtualLock` on Windows, `mlock` on Linux/macOS, both with a best-effort
attempt to raise the platform's default quota first, since either API simply
fails outright on a multi-hundred-MB range under the out-of-the-box limit.
Failure is silent and non-fatal: an unprivileged process keeps its weights
either way, just without a guarantee against a swap stall mid-decode under
memory pressure.

This is deliberately partial: the per-layer weights are not locked (that
would mean walking every tensor across every layer, real additional work, not
done here), only the embedding/output head. Enable with `DESIREEIA_MLOCK=1`.

## Hardware backends

Backends are probed and the strongest available one is selected automatically, in priority order:

1. NPU acceleration (vendor SDK)
2. NVIDIA CUDA
3. Intel Arc (Level Zero)
4. Metal (Apple Silicon)
5. CPU (always available — scalar fallback if AVX2 isn't present)

## Requirements

- .NET 10 SDK
- C++17 compiler and CMake for the native core (MSVC on Windows, clang on macOS, gcc/clang on Linux)
- AVX2-capable CPU for the accelerated 64-bit build (scalar fallback otherwise)

The managed GGUF reader and inspector are pure .NET and run without a native build on any platform.

## Building

```powershell
# native core
cmake -S src/DesireeIA.LocaleEngine -B native/out -DCMAKE_BUILD_TYPE=Release
cmake --build native/out --config Release

# .NET library
dotnet build DesireeIA.slnx
```

## Testing

```powershell
dotnet test DesireeIA.slnx
```

.NET tests detect whether the native library is present and skip automatically if it isn't.

## Tools

Model inspector (pure .NET, no native build required):

```powershell
dotnet run --project tools/DesireeIA.Inspect -- <model.gguf> [--tensors]
```

CLI for end-to-end testing and benchmarking:

```powershell
dotnet run --project cli/DesireeIA.Cli -- hw
dotnet run --project cli/DesireeIA.Cli -- info <model.gguf>
dotnet run --project cli/DesireeIA.Cli -- tokenize <model.gguf> "<text>"
dotnet run --project cli/DesireeIA.Cli -- generate <model.gguf> "<text>" [--max-tokens N] [--chat] [--temp T] [--top-k K] [--top-p P]
dotnet run --project cli/DesireeIA.Cli -- embed <model.gguf> "<text>"   (BERT encoders only)
dotnet run --project cli/DesireeIA.Cli -- bench <model.gguf> [--tokens N] [--warmup N] [--prompt "<text>"]



EXAMPLE FOR TEST IN WINDOWS CMD
C:\Sorgenti\Personal\DesireeIA\cli\DesireeIA.Cli\bin\Release\net10.0\desireeia-cli.exe chat "C:\Users\fpassaro\AppData\Local\Kodinn\google_gemma-3-4b-it-Q4_K_M.gguf" --temp 0.7 --top-k 40 --top-p 0.9 --max-tokens 2048
```

## License
PREAMBLE & VISION
Technology should empower people and drive progress. The creator of this project, 
Passaro Francesco Paolo, is strongly open to collaborations and ideas from developers, 
researchers, and innovators worldwide. Together, through open dialogue, quality code, 
and shared vision, we can make the world a better place.

1. PERMITTED USES
Subject to the terms of this License, the Author (Passaro Francesco Paolo) grants you 
a non-exclusive, worldwide, royalty-free license to:
   a) Download, install, execute, and run this software for both personal and commercial purposes.
   b) Copy, duplicate, and share the original, unmodified source code or binaries with others, 
      provided that this copyright notice and license remain intact.

2. RESTRICTIONS
To protect the integrity and vision of the core engine, the following restrictions apply:
   a) NO MODIFICATION: You may not alter, modify, patch, adapt, or create derivative works 
      from this source code or binaries without explicit written permission from Passaro Francesco Paolo.
   b) NO UNAUTHORIZED INTEGRATION: You may not extract or incorporate parts of this source code 
      into other projects without prior written approval.
   c) NO AI AGENT INGESTION, TRAINING OR REPLICATION: Artificial Intelligence systems, AI agents, 
      automated crawlers, machine learning models, or LLMs are strictly prohibited from reading, 
      ingesting, scraping, parsing, or analyzing this source code or binaries for the purpose of 
      machine learning training, fine-tuning, code duplication, or generating derivative code 
      without explicit, prior written consent from Passaro Francesco Paolo.

3. COLLABORATIONS & CONTRIBUTIONS
If you wish to propose improvements, modify the engine, integrate it into a new architecture, 
or collaborate on future developments, you are warmly invited to contact the Author. 
Pull requests, ideas, and partnerships are welcome upon review and approval by Passaro Francesco Paolo.

4. DISCLAIMER OF WARRANTY
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. 
IN NO EVENT SHALL THE AUTHOR (PASSARO FRANCESCO PAOLO) BE LIABLE FOR ANY CLAIM, DAMAGES, 
OR OTHER LIABILITY ARISING FROM THE USE OF THE SOFTWARE.
