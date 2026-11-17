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
/// The engine's quantized KV cache keeps only a window of positions in the local-attention layers: a model with 27 local
/// layers out of 36 needs the full context for 9 layers only. The context that fits the video memory follows.
/// </summary>
public class WindowedKvFitTests
{
    private const long Mb = 1024 * 1024;

    // Spark-X2.6-4B: 36 layers, 4 KV heads, key+value 256 each, window 512, 27 local layers.
    private static ModelTraits Spark(int locals = 27, long bytes = 3_543_441_312) =>
        new("spark2_5", 1_048_576, LayerCount: 36, HeadCountKv: 4, KeyLength: 256, ValueLength: 256,
            SlidingWindow: 512, FileSizeBytes: bytes,
            RecommendedTemperature: null, RecommendedTopK: null, RecommendedTopP: null)
        { LocalLayerCount = locals };

    [Fact]
    public void QuantizedCache_OnlyTheGlobalLayersGrowWithTheContext()
    {
        var m = Spark();
        ulong whole = m.KvBytesPerTokenFor(quantized: true);              // every layer whole: the old upper bound
        Assert.Equal(78_336UL, whole);
        ulong at16k = m.KvBytesFor(true, 16384), at64k = m.KvBytesFor(true, 65536);
        // 9 global layers grow 4x, the 27 local ones stay at their fixed size
        ulong localFixed = 27UL * 1599 * (whole / 36);
        Assert.Equal(localFixed + 9UL * 16384 * (whole / 36), at16k);
        Assert.Equal(localFixed + 9UL * 65536 * (whole / 36), at64k);
        Assert.True(at64k < whole * 65536 / 3);
    }

    [Fact]
    public void FloatCache_KeepsEveryLayerWhole() =>
        Assert.Equal(Spark().KvBytesPerTokenFor(false) * 4096, Spark().KvBytesFor(false, 4096));

    [Fact]
    public void ModelWithoutLocalLayers_IsUnchanged() =>
        Assert.Equal(Spark(locals: 0).KvBytesPerTokenFor(true) * 10_000, Spark(locals: 0).KvBytesFor(true, 10_000));

    [Fact]
    public void MaxPositions_InvertsTheSize()
    {
        var m = Spark();
        foreach (long budgetMb in new long[] { 400, 900, 1500, 6000 })
        {
            long p = m.MaxPositionsFor(true, (ulong)(budgetMb * Mb));
            Assert.True(m.KvBytesFor(true, p) <= (ulong)(budgetMb * Mb));
            Assert.True(m.KvBytesFor(true, p + 64) > (ulong)(budgetMb * Mb));
        }
        // the 1.5 GB a 6 GB card leaves hold ~75k positions, not the 20k a whole-layer count gave
        Assert.InRange(m.MaxPositionsFor(true, (ulong)(1500 * Mb)), 70_000, 80_000);
    }

    [Fact]
    public void SixGigabyteCard_NowFitsAContextWellPastTheTarget()
    {
        var hw = new HardwareProfile(CpuThreads: 16, Avx: true, Avx2: true, Avx512: false, Neon: false, CudaDeviceCount: 1,
            RamTotalMb: 32_000, RamFreeMb: 24_000, IntelGpuCount: 0, AxeleraDeviceCount: 0, Metal: false, Vulkan: false,
            Gpus: new[] { new GpuInfo(0, "Test GPU", 6141 * Mb, 5080 * Mb, 8, 9, 24) });
        var plan = new ExecutionPlan { Backend = InferenceBackend.Cuda, ThreadCount = 16, RamBudgetMb = 24_000, KvCompression = true };
        var c = AutoConfigurator.Compute(hw, plan, Spark());
        Assert.Equal(AutoConfigurator.WindowedTargetContextSize, c.ContextSize);
        Assert.InRange(c.MaxContextSize, 40_960, 131_072);
        // without KV compression the whole float cache is back and the windowed target does not apply
        var wholePlan = new ExecutionPlan { Backend = InferenceBackend.Cuda, ThreadCount = 16, RamBudgetMb = 24_000, KvCompression = false };
        var whole = AutoConfigurator.Compute(hw, wholePlan, Spark());
        Assert.True(whole.ContextSize <= AutoConfigurator.TargetContextSize);   // the float cache costs 4x: never the windowed target
    }
}
