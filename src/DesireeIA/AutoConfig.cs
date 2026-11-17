// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA.Formats;

namespace DesireeIA;

/// <summary>
/// What the auto-configuration needs to know about a model, read from the
/// GGUF header only (no weights are loaded).
/// </summary>
public sealed record ModelTraits(
    string Architecture,
    uint TrainedContextLength,
    uint LayerCount,
    uint HeadCountKv,
    uint KeyLength,
    uint ValueLength,
    uint SlidingWindow,
    long FileSizeBytes,
    float? RecommendedTemperature,
    int? RecommendedTopK,
    float? RecommendedTopP)
{
    /// <summary>Layers with local attention (they attend to <see cref="SlidingWindow"/> positions only).</summary>
    public int LocalLayerCount { get; init; }

    /// <summary>
    /// Positions beyond the window a local layer keeps in the engine's windowed cache: the rows behind the current one
    /// that a rolled-back draft may need (64) plus the block a long prompt goes through at a time (1024).
    /// </summary>
    public const int WindowedKvSlack = 64 + 1024;

    /// <summary>Positions one local layer holds in the windowed cache (a fixed size, whatever the context).</summary>
    public long LocalKvPositions => SlidingWindow > 0 ? SlidingWindow - 1 + WindowedKvSlack : 0;

    /// <summary>
    /// KV-cache bytes for <paramref name="positions"/> of context as the engine stores it. With the Q8_0 cache the local
    /// layers of a model keep only <see cref="LocalKvPositions"/> positions, so only the global layers grow with the
    /// context; the float32 cache keeps every layer whole.
    /// </summary>
    public ulong KvBytesFor(bool quantized, long positions)
    {
        if (positions <= 0 || LayerCount == 0) return 0;
        ulong perLayer = HeadCountKv * (ulong)(KeyLength + ValueLength);
        ulong perLayerBytes = quantized ? (perLayer * 34 + 31) / 32 : perLayer * 4;
        ulong locals = quantized && SlidingWindow > 0 ? (ulong)Math.Clamp(LocalLayerCount, 0, (int)LayerCount) : 0;
        ulong globals = LayerCount - locals;
        ulong localPositions = (ulong)Math.Min(positions, LocalKvPositions);
        return globals * (ulong)positions * perLayerBytes + locals * localPositions * perLayerBytes;
    }

    /// <summary>The context (positions) a KV cache of at most <paramref name="budgetBytes"/> holds.</summary>
    public long MaxPositionsFor(bool quantized, ulong budgetBytes)
    {
        ulong whole = KvBytesFor(quantized, 1);
        if (whole == 0) return 0;
        ulong locals = quantized && SlidingWindow > 0 ? (ulong)Math.Clamp(LocalLayerCount, 0, (int)LayerCount) : 0;
        if (locals == 0) return (long)Math.Min(budgetBytes / whole, (ulong)int.MaxValue);
        ulong perLayerBytes = whole / LayerCount;
        ulong fixedBytes = locals * (ulong)LocalKvPositions * perLayerBytes;
        ulong globals = LayerCount - locals;
        if (globals == 0) return int.MaxValue;
        if (budgetBytes > fixedBytes)
        {
            ulong byGlobal = (budgetBytes - fixedBytes) / (globals * perLayerBytes);
            if (byGlobal >= (ulong)LocalKvPositions) return (long)Math.Min(byGlobal, (ulong)int.MaxValue);
        }
        return (long)Math.Min(budgetBytes / whole, (ulong)LocalKvPositions);   // small: every layer holds every position
    }

    /// <summary>
    /// KV-cache bytes per context token with the float32 cache (keys + values,
    /// every layer), as the engine stores it when KV compression is off. An
    /// upper bound: sliding-window layers and latent attention (DeepSeek MLA)
    /// need less, so budgets computed from it err on the safe side.
    /// </summary>
    public ulong KvBytesPerToken => KvBytesPerTokenFor(quantized: false);

    /// <summary>
    /// KV-cache bytes per context token as the engine actually stores it:
    /// float32 (4 bytes per value), or Q8_0 when
    /// <see cref="ExecutionPlan.KvCompression"/> is on (34 bytes per 32 values).
    /// </summary>
    public ulong KvBytesPerTokenFor(bool quantized)
    {
        ulong values = (ulong)LayerCount * HeadCountKv * (KeyLength + ValueLength);
        return quantized ? (values * 34 + 31) / 32 : values * 4;
    }

    /// <summary>Reads the traits of a GGUF model file.</summary>
    public static ModelTraits Read(string modelPath)
    {
        var reader = new GgufReader();
        var meta = reader.Open(modelPath);
        var kv = reader.RawKeyValues;
        var arch = meta.Architecture;

        uint headCount = U(kv, $"{arch}.attention.head_count") ?? 0;
        uint headCountKv = U(kv, $"{arch}.attention.head_count_kv") ?? headCount;
        // Without explicit key/value lengths the head size is embedding / heads.
        uint headDim = headCount > 0 ? meta.EmbeddingSize / headCount : 0;
        uint keyLength = U(kv, $"{arch}.attention.key_length") ?? headDim;
        uint valueLength = U(kv, $"{arch}.attention.value_length") ?? keyLength;

        // general.sampling.* are the author's suggested values (GGUF spec).
        // top_k <= 0 means "disabled" there, so it's reported as no suggestion.
        int? topK = I(kv, "general.sampling.top_k");
        uint slidingWindow = U(kv, $"{arch}.attention.sliding_window") ?? 0;
        return new ModelTraits(
            Architecture: arch,
            TrainedContextLength: meta.ContextLength,
            LayerCount: meta.LayerCount,
            HeadCountKv: headCountKv,
            KeyLength: keyLength,
            ValueLength: valueLength,
            SlidingWindow: slidingWindow,
            FileSizeBytes: new FileInfo(modelPath).Length,
            RecommendedTemperature: F(kv, "general.sampling.temp"),
            RecommendedTopK: topK is > 0 ? topK : null,
            RecommendedTopP: F(kv, "general.sampling.top_p"))
        { LocalLayerCount = CountLocalLayers(kv, arch, meta.LayerCount, slidingWindow) };
    }

    /// <summary>Layers with local attention: from the per-layer flags, or a period (one global layer per period, 6 by default).</summary>
    private static int CountLocalLayers(IReadOnlyDictionary<string, object> kv, string arch, uint layers, uint slidingWindow)
    {
        if (slidingWindow == 0 || layers == 0) return 0;
        var key = $"{arch}.attention.sliding_window_pattern";
        if (kv.TryGetValue(key, out var v) && v is GgufReader.ArrayMarker { Values: { Count: > 0 } flags })
            return flags.Take((int)layers).Count(x => x != 0);
        uint period = U(kv, key) ?? 6;
        if (period == 0) return 0;
        int n = 0;
        for (uint l = 0; l < layers; l++) if (l % period != period - 1) n++;
        return n;
    }

    private static uint? U(IReadOnlyDictionary<string, object> kv, string key) =>
        kv.TryGetValue(key, out var v) ? v switch
        {
            uint u => u,
            int i when i >= 0 => (uint)i,
            ulong ul when ul <= uint.MaxValue => (uint)ul,
            long l when l >= 0 && l <= uint.MaxValue => (uint)l,
            ushort us => us,
            byte b => b,
            _ => null
        } : null;

    private static int? I(IReadOnlyDictionary<string, object> kv, string key) =>
        kv.TryGetValue(key, out var v) ? v switch
        {
            int i => i,
            uint u when u <= int.MaxValue => (int)u,
            long l when l >= int.MinValue && l <= int.MaxValue => (int)l,
            ulong ul when ul <= int.MaxValue => (int)ul,
            _ => null
        } : null;

    private static float? F(IReadOnlyDictionary<string, object> kv, string key) =>
        kv.TryGetValue(key, out var v) ? v switch
        {
            float f => f,
            double d => (float)d,
            _ => null
        } : null;
}

/// <summary>
/// A complete, ready-to-use configuration for a model on THIS machine:
/// execution plan (backend, threads, RAM), sampling, context and output
/// length. <see cref="Notes"/> explains every choice in plain words.
/// </summary>
public sealed record AutoConfiguration(
    HardwareProfile Hardware,
    ExecutionPlan Plan,
    ModelTraits Model,
    SamplingOptions Sampling,
    int ContextSize,
    int MaxTokens,
    IReadOnlyList<string> Notes,
    // The largest context that still fits this machine (trained length, RAM and, on a GPU, VIDEO MEMORY): what a caller
    // may grow the context to when a prompt needs more than ContextSize. Never below ContextSize; 0 when not computed.
    int MaxContextSize = 0);

/// <summary>
/// Picks the parameters a model should run with on the current hardware, so
/// that loading a model "just works" without hand-tuning:
/// <list type="bullet">
/// <item>backend, threads and RAM budget come from the native planner
/// (<see cref="DesireeIAEngine.BuildPlan"/>), which probes CPU features,
/// CUDA/Intel/Metal devices and free memory;</item>
/// <item>sampling is never greedy: greedy decoding falls into repetitive loops
/// on long generations, the main cause of "the local model repeats itself";</item>
/// <item>the context is as large as the target allows, but never beyond what
/// the model was trained on nor beyond what the KV cache can fit in memory;</item>
/// <item>the output length is bounded by the context, so a reply can never
/// crowd out the prompt.</item>
/// </list>
/// </summary>
public static class AutoConfigurator
{
    /// <summary>Default temperature: varied but focused text.</summary>
    public const float DefaultTemperature = 0.7f;
    public const int DefaultTopK = 40;
    public const float DefaultTopP = 0.9f;
    /// <summary>Longest reply by default.</summary>
    public const int DefaultMaxTokens = 2048;
    /// <summary>Context to aim for: room for a system prompt with tool
    /// descriptions, a conversation and a full-length reply.</summary>
    public const int TargetContextSize = 16384;
    /// <summary>
    /// Context to aim for when the KV cache is windowed (Q8_0 cache, a model with local-attention layers): only the global
    /// layers grow with it, so twice the context costs about what 16384 positions cost with the whole cache.
    /// </summary>
    public const int WindowedTargetContextSize = 32768;

    /// <summary>The context to aim for with this model and plan: <see cref="WindowedTargetContextSize"/> when the cache is windowed.</summary>
    public static int TargetContextFor(ModelTraits model, ExecutionPlan plan) =>
        plan.KvCompression && model.LocalLayerCount > 0 && model.SlidingWindow > 0 ? WindowedTargetContextSize : TargetContextSize;
    /// <summary>Smallest context worth running with.</summary>
    public const int MinContextSize = 2048;
    /// <summary>Memory always left to the OS and the rest of the process.</summary>
    public const ulong SafetyMarginMb = 1024;
    /// <summary>
    /// Video memory kept free after the weights and the KV cache, for the compute buffers of a prefill. Smaller than
    /// <see cref="SafetyMarginMb"/> on purpose: the free figure of the driver already excludes what the desktop uses.
    /// MEASURED with a 4B model on a card showing 5080 MB free: weights 3379 MB + KV 1224 MB (16384 tokens) left ~480 MB
    /// and ran at 2100 tokens/s.
    /// </summary>
    public const ulong VideoMarginMb = 384;

    /// <summary>
    /// Configures <paramref name="modelPath"/> for the current machine.
    /// <paramref name="overrides"/> works as in <see cref="DesireeIAEngine.BuildPlan"/>.
    /// </summary>
    public static AutoConfiguration Configure(string modelPath, ExecutionPlan? overrides = null,
                                              int targetContext = TargetContextSize)
    {
        var traits = ModelTraits.Read(modelPath);
        var hw = DesireeIAEngine.DetectHardware();
        var plan = DesireeIAEngine.BuildPlan(modelPath, overrides);
        return Compute(hw, plan, traits, targetContext);
    }

    /// <summary>
    /// The context that fits the VIDEO MEMORY of the GPU the model runs on, or null when it cannot be told (not a CUDA
    /// plan, no GPU probed, KV size unknown). Weights are counted in video memory, then the KV cache per token, with
    /// <see cref="VideoMarginMb"/> left for the compute buffers. MEASURED on a 6 GB card with a 4B
    /// model (3.5 GB of weights, 78 KB of KV per token): 16384 tokens run, 40960 spill into shared memory and the
    /// prefill gets seven times slower with the GPU pinned at 100%.
    /// </summary>
    public static int? MaxContextByVideoMemory(HardwareProfile hw, ExecutionPlan plan, ModelTraits model, out string? note)
    {
        note = null;
        if (plan.Backend != InferenceBackend.Cuda || hw.Gpus is not { Count: > 0 } gpus) return null;
        ulong kvPerToken = model.KvBytesPerTokenFor(plan.KvCompression);
        if (kvPerToken == 0) return null;
        // the local layers of a windowed cache stay small: only the global ones grow with the context
        ulong kvAt64k = model.KvBytesFor(plan.KvCompression, 65536) / 65536;

        var gpu = gpus.OrderByDescending(g => g.FreeBytes).First();
        long freeMb = gpu.FreeBytes / (1024 * 1024);
        long modelMb = Math.Max(0, model.FileSizeBytes) / (1024 * 1024);
        long usableMb = freeMb - (long)VideoMarginMb - modelMb;
        if (usableMb <= 0)
        {
            note = $"video memory: {freeMb} MB free on {gpu.Name}, the weights alone are ~{modelMb} MB (+{VideoMarginMb} MB margin): " +
                   "no room for a KV cache on the GPU, the smallest context is used";
            return MinContextSize;
        }
        long byVideo = model.MaxPositionsFor(plan.KvCompression, (ulong)usableMb * 1024 * 1024);
        int fit = (int)Math.Min(byVideo, int.MaxValue);
        note = $"video memory: {freeMb} MB free on {gpu.Name}, weights ~{modelMb} MB, ~{kvAt64k / 1024} KB of KV per token: " +
               $"up to {fit} tokens of context fit";
        return fit;
    }

    /// <summary>The pure decision, separated from probing so it can be tested.</summary>
    public static AutoConfiguration Compute(HardwareProfile hw, ExecutionPlan plan, ModelTraits model,
                                            int targetContext = TargetContextSize)
    {
        var notes = new List<string>
        {
            $"backend {plan.Backend}, {plan.ThreadCount} threads, RAM budget {plan.RamBudgetMb} MB " +
            $"(cpu threads {hw.CpuThreads}, cuda devices {hw.CudaDeviceCount}, free RAM {hw.RamFreeMb} MB)"
        };

        // ── Context ──────────────────────────────────────────────────────
        if (targetContext == TargetContextSize) targetContext = TargetContextFor(model, plan);   // the default target follows the cache layout
        int context = Math.Max(MinContextSize, targetContext);
        int maxContext = int.MaxValue;   // what could still fit, whatever the target
        if (model.TrainedContextLength > 0)
        {
            maxContext = (int)Math.Min(maxContext, model.TrainedContextLength);
            if (model.TrainedContextLength < context)
            {
                context = (int)model.TrainedContextLength;
                notes.Add($"context limited to the trained length {model.TrainedContextLength}");
            }
        }

        ulong budgetMb = plan.RamBudgetMb > 0 ? plan.RamBudgetMb : hw.RamFreeMb;
        ulong modelMb = (ulong)Math.Max(0, model.FileSizeBytes) / (1024 * 1024);
        ulong kvPerToken = model.KvBytesPerTokenFor(plan.KvCompression);
        // ON A GPU THE KV CACHE LIVES IN VIDEO MEMORY: free RAM says nothing about it. MEASURED with 4 GB of free RAM and
        // 5 GB of free video memory, the RAM test planned 2,048 tokens where 17,628 fit the card. The video fit below decides.
        bool kvInVideo = MaxContextByVideoMemory(hw, plan, model, out _) is not null;
        if (budgetMb > 0 && kvPerToken > 0 && !kvInVideo)
        {
            // Weights may sit in VRAM on GPU backends, but VRAM size isn't
            // probed: counting them against RAM keeps the estimate safe.
            ulong freeForKvMb = budgetMb > modelMb + SafetyMarginMb ? budgetMb - modelMb - SafetyMarginMb : 0;
            // Half of what is left: the KV cache is not the only buffer.
            ulong byMemory = (ulong)model.MaxPositionsFor(plan.KvCompression, freeForKvMb * 1024 * 1024 / 2);
            maxContext = (int)Math.Min((ulong)maxContext, byMemory);
            if (byMemory < (ulong)context)
            {
                context = (int)Math.Max((ulong)MinContextSize, byMemory);
                notes.Add($"context limited by memory: ~{kvPerToken / 1024} KB of KV cache per token " +
                          $"({(plan.KvCompression ? "Q8_0" : "float32")}), " +
                          $"{freeForKvMb} MB available after the model");
            }
        }

        // The video memory of the GPU: the RAM budget above does not see it, and a context that does not fit it is not
        // refused by the driver, it is silently spilled into shared memory and every prefill crawls.
        if (MaxContextByVideoMemory(hw, plan, model, out var videoNote) is { } byVideo)
        {
            notes.Add(videoNote!);
            maxContext = Math.Min(maxContext, byVideo);
            if (byVideo < context)
            {
                context = Math.Max(MinContextSize, byVideo);
                notes.Add($"context limited by video memory to {context} tokens");
            }
        }
        if (context > MinContextSize) context -= context % 1024;
        context = Math.Max(MinContextSize, context);
        if (maxContext < int.MaxValue && maxContext > MinContextSize) maxContext -= maxContext % 1024;
        maxContext = maxContext == int.MaxValue ? context : Math.Max(context, maxContext);

        // ── Output length ────────────────────────────────────────────────
        int maxTokens = Math.Min(DefaultMaxTokens, context / 4);

        // ── Sampling ─────────────────────────────────────────────────────
        var sampling = new SamplingOptions
        {
            Temperature = DefaultTemperature,
            TopK = DefaultTopK,
            TopP = DefaultTopP,
        };
        if (model.RecommendedTemperature is not null || model.RecommendedTopP is not null || model.RecommendedTopK is not null)
        {
            notes.Add("the model file suggests " +
                      $"temp={Fmt(model.RecommendedTemperature)} top-k={(model.RecommendedTopK?.ToString() ?? "-")} " +
                      $"top-p={Fmt(model.RecommendedTopP)}; defaults used: " +
                      $"temp={DefaultTemperature} top-k={DefaultTopK} top-p={DefaultTopP}");
        }

        notes.Add($"context {context} tokens, replies up to {maxTokens} tokens");
        return new AutoConfiguration(hw, plan, model, sampling, context, maxTokens, notes, maxContext);
    }

    private static string Fmt(float? f) =>
        f?.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture) ?? "-";
}
