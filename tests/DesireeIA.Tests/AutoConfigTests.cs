// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

public class AutoConfigTests
{
    private static HardwareProfile Hw(ulong freeMb = 11_000) =>
        new(CpuThreads: 16, Avx: true, Avx2: true, Avx512: false, Neon: false, CudaDeviceCount: 1,
            RamTotalMb: 32_000, RamFreeMb: freeMb, IntelGpuCount: 0, AxeleraDeviceCount: 0,
            Metal: false, Vulkan: false);

    private static ExecutionPlan Plan(ulong ramMb = 11_000) =>
        new() { Backend = InferenceBackend.Cuda, ThreadCount = 16, RamBudgetMb = ramMb };

    // Numbers from a real Spark-X2.5-4B-Q4_K_M.gguf header.
    private static ModelTraits Spark(uint trained = 1_048_576, long bytes = 2_600_224_352) =>
        new("spark2_5", trained, LayerCount: 36, HeadCountKv: 4, KeyLength: 256, ValueLength: 256,
            SlidingWindow: 512, FileSizeBytes: bytes,
            RecommendedTemperature: 1.0f, RecommendedTopK: null, RecommendedTopP: 0.95f);

    [Fact]
    public void Sampling_IsNeverGreedy()
    {
        var c = AutoConfigurator.Compute(Hw(), Plan(), Spark());
        Assert.Equal(AutoConfigurator.DefaultTemperature, c.Sampling.Temperature);
        Assert.Equal(AutoConfigurator.DefaultTopK, c.Sampling.TopK);
        Assert.Equal(AutoConfigurator.DefaultTopP, c.Sampling.TopP);
        Assert.True(c.Sampling.Temperature > 0);
    }

    [Fact]
    public void ComfortableMachine_GetsTargetContextAndFullReplies()
    {
        var c = AutoConfigurator.Compute(Hw(), Plan(), Spark());
        Assert.Equal(AutoConfigurator.TargetContextSize, c.ContextSize);
        Assert.Equal(AutoConfigurator.DefaultMaxTokens, c.MaxTokens);
    }

    [Fact]
    public void Context_NeverExceedsTrainedLength()
    {
        var c = AutoConfigurator.Compute(Hw(), Plan(), Spark(trained: 4096));
        Assert.Equal(4096, c.ContextSize);
        Assert.Equal(1024, c.MaxTokens);   // a reply can't take more than a quarter
    }

    [Fact]
    public void TightMemory_ShrinksContext_ButNotBelowMinimum()
    {
        // Model almost as big as the budget: almost nothing left for the KV cache.
        var c = AutoConfigurator.Compute(Hw(3_000), Plan(3_000), Spark());
        Assert.Equal(AutoConfigurator.MinContextSize, c.ContextSize);
        Assert.Contains(c.Notes, n => n.Contains("memory"));
    }

    [Fact]
    public void ModerateMemory_ContextIsBoundedAndAligned()
    {
        // ~6 GB budget: 2.5 GB model + 1 GB margin, half of the rest for ~144 KB/token.
        var c = AutoConfigurator.Compute(Hw(6_000), Plan(6_000), Spark());
        Assert.InRange(c.ContextSize, AutoConfigurator.MinContextSize, AutoConfigurator.TargetContextSize - 1);
        Assert.Equal(0, c.ContextSize % 1024);
    }

    [Fact]
    public void ModelSuggestion_IsReportedInNotes()
    {
        var c = AutoConfigurator.Compute(Hw(), Plan(), Spark());
        Assert.Contains(c.Notes, n => n.Contains("suggests"));
    }

    [Fact]
    public void KvBytesPerToken_MatchesHeaderArithmetic()
    {
        // 2 bytes * 36 layers * 4 kv heads * (256 + 256)
        Assert.Equal(147_456UL, Spark().KvBytesPerToken);
    }

    [Fact]
    public void ModelTraits_ReadsRealModel_WhenAvailable()
    {
        var path = Environment.GetEnvironmentVariable("DESIREEIA_TEST_MODEL_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // real model optional, like the other tests
        var t = ModelTraits.Read(path);
        Assert.False(string.IsNullOrEmpty(t.Architecture));
        Assert.True(t.LayerCount > 0);
        Assert.True(t.KvBytesPerToken > 0);
    }
}
