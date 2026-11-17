// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

/// <summary>
/// The context follows the VIDEO MEMORY of the GPU, not just the RAM.
/// MEASURED on a 6 GB laptop card with a 4B model (3.5 GB of weights, 78 KB of KV cache per token): a 9174-token prefill
/// takes 15.7 s at a context of 16384; at 40960 the KV cache and the weights (6.7 GB) overflow the card, the first layer
/// alone takes 80 s and the GPU stays at 100%.
/// </summary>
public class VideoMemoryFitTests
{
    private const long Mb = 1024 * 1024;

    private static HardwareProfile Hw(long gpuTotalMb, long gpuFreeMb, ulong ramFreeMb = 24_000) =>
        new(CpuThreads: 16, Avx: true, Avx2: true, Avx512: false, Neon: false, CudaDeviceCount: 1,
            RamTotalMb: 32_000, RamFreeMb: ramFreeMb, IntelGpuCount: 0, AxeleraDeviceCount: 0,
            Metal: false, Vulkan: false,
            Gpus: new[] { new GpuInfo(0, "Test GPU", gpuTotalMb * Mb, gpuFreeMb * Mb, 8, 9, 24) });

    private static ExecutionPlan Cuda(bool kvCompression = true) =>
        new() { Backend = InferenceBackend.Cuda, ThreadCount = 16, RamBudgetMb = 24_000, KvCompression = kvCompression };

    // Spark-X2.6-4B-Q4_K_M: 36 layers, 8 KV heads, key+value 256 each -> 78,336 bytes of Q8_0 KV per token.
    private static ModelTraits Spark26(long bytes = 3_543_441_312) =>
        new("spark2_6", 1_048_576, LayerCount: 36, HeadCountKv: 8, KeyLength: 128, ValueLength: 128,
            SlidingWindow: 0, FileSizeBytes: bytes,
            RecommendedTemperature: null, RecommendedTopK: null, RecommendedTopP: null);

    [Fact]
    public void KvBytesPerToken_MatchesWhatTheEngineWrites() =>
        Assert.Equal(78_336UL, Spark26().KvBytesPerTokenFor(quantized: true));

    [Fact]
    public void SixGigabyteCard_FitsTheTargetButNotMuchMore()
    {
        // 6141 MB card with the desktop using ~330 MB.
        var c = AutoConfigurator.Compute(Hw(6141, 5800), Cuda(), Spark26());
        Assert.Equal(AutoConfigurator.TargetContextSize, c.ContextSize);
        Assert.InRange(c.MaxContextSize, 16384, 40959);
        Assert.Contains(c.Notes, n => n.Contains("video memory", StringComparison.Ordinal));
        // 40960 tokens (3.2 GB of KV) never fit this card: the ceiling is well below it.
        Assert.True(c.MaxContextSize < 40960);
    }

    [Fact]
    public void TheRealCard_5080MbFree_RunsTheTargetContext()
    {
        // What the driver reported with the desktop running: 5080 MB free of 6140. MEASURED: 16384 tokens ran at
        // ~2100 tokens/s on it, so the fit must not refuse them (a 1 GB margin chose 8192 and starved the prompt).
        var c = AutoConfigurator.Compute(Hw(6140, 5080), Cuda(), Spark26());
        Assert.Equal(AutoConfigurator.TargetContextSize, c.ContextSize);
        Assert.InRange(c.MaxContextSize, 16384, 18432);
    }

    [Fact]
    public void LowFreeRam_DoesNotShrinkTheContext_WhenTheKvLivesInVideoMemory()
    {
        // MEASURED: 4 GB of free RAM left "0 MB after the model" and planned 2048 tokens, while 17,628 fit the card.
        var plan = new ExecutionPlan { Backend = InferenceBackend.Cuda, ThreadCount = 16, RamBudgetMb = 4041, KvCompression = true };
        var c = AutoConfigurator.Compute(Hw(6140, 5080, ramFreeMb: 4042), plan, Spark26());
        Assert.Equal(AutoConfigurator.TargetContextSize, c.ContextSize);
    }

    [Fact]
    public void BigCard_AllowsALargerContext_WhenAskedFor()
    {
        var c = AutoConfigurator.Compute(Hw(24_576, 23_000), Cuda(), Spark26(), targetContext: 65_536);
        Assert.Equal(65_536, c.ContextSize);
        Assert.True(c.MaxContextSize >= 65_536);
    }

    [Fact]
    public void BigCard_KeepsTheDefaultTarget_ButReportsHowFarItCanGrow()
    {
        var c = AutoConfigurator.Compute(Hw(24_576, 23_000), Cuda(), Spark26());
        Assert.Equal(AutoConfigurator.TargetContextSize, c.ContextSize);
        Assert.True(c.MaxContextSize > AutoConfigurator.TargetContextSize);
    }

    [Fact]
    public void ACardTooSmallForTheWeights_FallsBackToTheSmallestContext()
    {
        var c = AutoConfigurator.Compute(Hw(4096, 3500), Cuda(), Spark26());
        Assert.Equal(AutoConfigurator.MinContextSize, c.ContextSize);
        Assert.Contains(c.Notes, n => n.Contains("weights alone", StringComparison.Ordinal));
    }

    [Fact]
    public void WithoutAGpuProbe_BehavesAsBefore()
    {
        var hw = Hw(6141, 5800) with { Gpus = null };
        var c = AutoConfigurator.Compute(hw, Cuda(), Spark26());
        Assert.Equal(AutoConfigurator.TargetContextSize, c.ContextSize);
        Assert.DoesNotContain(c.Notes, n => n.Contains("video memory", StringComparison.Ordinal));
    }

    [Fact]
    public void ACpuPlan_IgnoresTheVideoMemory()
    {
        var plan = new ExecutionPlan { Backend = InferenceBackend.Cpu, ThreadCount = 16, RamBudgetMb = 24_000 };
        Assert.Null(AutoConfigurator.MaxContextByVideoMemory(Hw(6141, 5800), plan, Spark26(), out _));
    }

    [Fact]
    public void TheMostFreeGpuIsTheOneCounted()
    {
        var hw = Hw(6141, 1000) with
        {
            Gpus = new[]
            {
                new GpuInfo(0, "Small", 6141 * Mb, 1000 * Mb, 8, 9, 24),
                new GpuInfo(1, "Big", 24_576 * Mb, 23_000 * Mb, 8, 9, 80),
            }
        };
        var fit = AutoConfigurator.MaxContextByVideoMemory(hw, Cuda(), Spark26(), out var note);
        Assert.True(fit > 100_000);
        Assert.Contains("Big", note!, StringComparison.Ordinal);
    }

    [Fact]
    public void MemoryBreakdown_PrintsOneLine()
    {
        var b = new MemoryBreakdown("RTX", 6141, 1300, 3379, 1224, 16384, 78_336);
        Assert.Equal(4841, b.VideoUsedMb);
        Assert.Contains("KV ~1224 MB for 16384 positions (76 KB/token)", b.ToString(), StringComparison.Ordinal);
    }
}
