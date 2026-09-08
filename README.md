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

1. Detects hardware (CPU core count, AVX/AVX2/AVX512/NEON, NUMA, CUDA/Intel Arc/Metal/NPU acceleration, RAM).
2. Builds an execution plan automatically unless the caller overrides it (RAM budget, thread count, quantization, format).
3. Applies tiered caching for MoE experts:
   - experts streamed from disk with an LRU cache,
   - a learned hot-set (frequently used experts pinned, persisted alongside the model),
   - one-layer-ahead expert prefetch,
   - batch-union expert fetching (each unique expert read once per batch of positions),
   - optional dual-SSD mirroring,
   - KV cache compression.

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
```

## License

Not yet decided.
