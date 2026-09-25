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
    /// <summary>
    /// Estimated KV-cache bytes per context token (keys + values, 16-bit, every
    /// layer). An upper bound: sliding-window layers and latent/compressed
    /// attention (DeepSeek MLA) need less, so budgets computed from it err on
    /// the safe side.
    /// </summary>
    public ulong KvBytesPerToken => 2UL * LayerCount * HeadCountKv * (KeyLength + ValueLength);

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
        return new ModelTraits(
            Architecture: arch,
            TrainedContextLength: meta.ContextLength,
            LayerCount: meta.LayerCount,
            HeadCountKv: headCountKv,
            KeyLength: keyLength,
            ValueLength: valueLength,
            SlidingWindow: U(kv, $"{arch}.attention.sliding_window") ?? 0,
            FileSizeBytes: new FileInfo(modelPath).Length,
            RecommendedTemperature: F(kv, "general.sampling.temp"),
            RecommendedTopK: topK is > 0 ? topK : null,
            RecommendedTopP: F(kv, "general.sampling.top_p"));
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
    IReadOnlyList<string> Notes);

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
    /// <summary>Smallest context worth running with.</summary>
    public const int MinContextSize = 2048;
    /// <summary>Memory always left to the OS and the rest of the process.</summary>
    public const ulong SafetyMarginMb = 1024;

    /// <summary>
    /// Configures <paramref name="modelPath"/> for the current machine.
    /// <paramref name="overrides"/> works as in <see cref="DesireeIAEngine.BuildPlan"/>.
    /// </summary>
    public static AutoConfiguration Configure(string modelPath, ExecutionPlan? overrides = null)
    {
        var traits = ModelTraits.Read(modelPath);
        var hw = DesireeIAEngine.DetectHardware();
        var plan = DesireeIAEngine.BuildPlan(modelPath, overrides);
        return Compute(hw, plan, traits);
    }

    /// <summary>The pure decision, separated from probing so it can be tested.</summary>
    public static AutoConfiguration Compute(HardwareProfile hw, ExecutionPlan plan, ModelTraits model)
    {
        var notes = new List<string>
        {
            $"backend {plan.Backend}, {plan.ThreadCount} threads, RAM budget {plan.RamBudgetMb} MB " +
            $"(cpu threads {hw.CpuThreads}, cuda devices {hw.CudaDeviceCount}, free RAM {hw.RamFreeMb} MB)"
        };

        // ── Context ──────────────────────────────────────────────────────
        int context = TargetContextSize;
        if (model.TrainedContextLength > 0 && model.TrainedContextLength < context)
        {
            context = (int)model.TrainedContextLength;
            notes.Add($"context limited to the trained length {model.TrainedContextLength}");
        }

        ulong budgetMb = plan.RamBudgetMb > 0 ? plan.RamBudgetMb : hw.RamFreeMb;
        ulong modelMb = (ulong)Math.Max(0, model.FileSizeBytes) / (1024 * 1024);
        ulong kvPerToken = model.KvBytesPerToken;
        if (budgetMb > 0 && kvPerToken > 0)
        {
            // Weights may sit in VRAM on GPU backends, but VRAM size isn't
            // probed: counting them against RAM keeps the estimate safe.
            ulong freeForKvMb = budgetMb > modelMb + SafetyMarginMb ? budgetMb - modelMb - SafetyMarginMb : 0;
            // Half of what is left: the KV cache is not the only buffer.
            ulong byMemory = freeForKvMb * 1024 * 1024 / 2 / kvPerToken;
            if (byMemory < (ulong)context)
            {
                context = (int)Math.Max((ulong)MinContextSize, byMemory);
                notes.Add($"context limited by memory: ~{kvPerToken / 1024} KB of KV cache per token, " +
                          $"{freeForKvMb} MB available after the model");
            }
        }
        if (context > MinContextSize) context -= context % 1024;
        context = Math.Max(MinContextSize, context);

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
        return new AutoConfiguration(hw, plan, model, sampling, context, maxTokens, notes);
    }

    private static string Fmt(float? f) =>
        f?.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture) ?? "-";
}
