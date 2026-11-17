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
/// The video memory of a real model: reserved with a guard, reported, warmed up and GIVEN BACK on Dispose.
/// Needs a GPU and a model (DESIREEIA_TEST_MODEL_PATH); does nothing without them, like the other real-model tests.
/// </summary>
[Collection("native-model")]
public class VideoMemoryLifecycleTests
{
    private const long Mb = 1024 * 1024;

    private static string? ModelOnAGpu()
    {
        try
        {
            if (DesireeIAEngine.Gpus().Count == 0) return null;
        }
        catch (Exception) { return null; }
        var path = Environment.GetEnvironmentVariable("DESIREEIA_TEST_MODEL_PATH");
        return !string.IsNullOrEmpty(path) && File.Exists(path) ? path : null;
    }

    private static long FreeMb() => DesireeIAEngine.Gpus().Max(g => g.FreeBytes) / Mb;

    [Fact]
    public void ReserveIsGuarded_BreakdownIsReported_AndDisposeGivesTheVideoMemoryBack()
    {
        var path = ModelOnAGpu();
        if (path is null) return;

        var cfg = AutoConfigurator.Configure(path);
        if (cfg.Plan.Backend != InferenceBackend.Cuda) return;
        long freeBefore = FreeMb();

        long freeWithModel;
        using (var model = LocalModel.Load(path, cfg.Plan))
        {
            // The context the fit chose must be accepted and reported.
            Assert.True(model.ReserveContext(cfg.ContextSize), string.Join("; ", cfg.Notes));
            var b = model.GetMemoryBreakdown();
            Assert.Equal(cfg.ContextSize, b.ReservedPositions);
            Assert.True(b.WeightsMb > 0 && b.KvReservedMb > 0, b.ToString());

            // A context whose KV cache cannot fit the free video memory is refused BEFORE allocating, not spilled.
            var traits = ModelTraits.Read(path);
            long perToken = (long)traits.KvBytesPerTokenFor(cfg.Plan.KvCompression);
            long freeNow = DesireeIAEngine.Gpus().Max(g => g.FreeBytes);
            int hugeContext = (int)Math.Min(int.MaxValue / 2, freeNow / Math.Max(1, perToken) + cfg.ContextSize + 4096);
            Assert.False(model.CanReserve(hugeContext, out var why));
            Assert.False(string.IsNullOrEmpty(why));
            Assert.False(model.ReserveContext(hugeContext));

            // The first request must not pay for the CUDA start-up.
            var sw = System.Diagnostics.Stopwatch.StartNew();
            model.Warmup();
            Console.WriteLine($"warmup {sw.Elapsed.TotalSeconds:F1}s; {model.GetMemoryBreakdown()}");
            freeWithModel = FreeMb();
        }

        // Give the driver a moment to account for the freed buffers.
        long freeAfter = 0;
        for (int i = 0; i < 20; i++)
        {
            freeAfter = FreeMb();
            if (freeAfter >= freeBefore - 600) break;
            Thread.Sleep(250);
        }
        Console.WriteLine($"free video memory: before {freeBefore} MB, with the model {freeWithModel} MB, after Dispose {freeAfter} MB");
        Assert.True(freeWithModel < freeBefore - 1000, "the model should occupy video memory");
        Assert.True(freeAfter >= freeBefore - 600, $"Dispose left {freeBefore - freeAfter} MB of video memory taken");
    }
}
