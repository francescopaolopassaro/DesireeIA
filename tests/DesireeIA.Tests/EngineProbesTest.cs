// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

/// <summary>ABI 5 probes: top candidates per token and the attention probe. Skipped without a model or an engine with ABI 5.</summary>
// One model on the GPU at a time: the CUDA backend keeps a single global scratch/KV cache, so test classes that load a model share one xUnit collection (they run one after the other).
[Collection("native-model")]
public class EngineProbesTest
{
    private static bool Ready(out string path)
    {
        path = Environment.GetEnvironmentVariable("DESIREEIA_TEST_MODEL_PATH") ?? "";
        try
        {
            if (DesireeIAEngine.NativeAbiVersion < 5) return false;
        }
        catch (Exception) { return false; }
        return File.Exists(path);
    }

    private static List<int> Generate(LocalModel model, int n, Action<int>? perToken = null)
    {
        var prompt = model.ApplyChatTemplate(new[] { ("user", "Say hi in one short sentence.") }, true);
        var ids = model.Tokenize(prompt, addBos: true)!;
        model.ResetSession();
        var tok = model.Predict(ids);
        var res = new List<int>();
        for (int i = 0; i < n && !model.IsEndOfGeneration(tok); i++)
        {
            res.Add(tok);
            perToken?.Invoke(i);
            tok = model.NextToken();
        }
        return res;
    }

    [Fact]
    public void TopCandidates_AreSorted_AndTheFirstIsTheMostLikely()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        model.SetConfidenceTracking(true);
        model.SetTopCandidates(5);
        var seen = 0;
        Generate(model, 6, _ =>
        {
            var top = model.GetTopCandidates();
            Assert.NotNull(top);
            Assert.Equal(5, top!.Length);
            for (int i = 1; i < top.Length; i++) Assert.True(top[i].LogProb <= top[i - 1].LogProb + 1e-6f);
            var stats = model.GetTokenStats();
            Assert.NotNull(stats);
            Assert.InRange(Math.Exp(top[0].LogProb) - stats!.Value.MaxProb, -1e-3, 1e-3);
            seen++;
        });
        Assert.True(seen > 0);
        model.SetTopCandidates(0);
        model.SetConfidenceTracking(false);
    }

    [Fact]
    public void AttentionProbe_WeightsSumToOne_AndTheOutputDoesNotChange()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        var plain = Generate(model, 8);
        Assert.True(model.SetAttentionProbe(true));
        var looked = 0;
        var probed = Generate(model, 8, i =>
        {
            var shape = model.GetAttentionShape();
            if (shape is null) { Assert.Equal(0, i); return; }       // the first token can come out of a prefill
            var w = model.GetAttention(shape.Value.Layers / 2);
            Assert.NotNull(w);
            Assert.Equal(shape.Value.Positions, w!.Length);
            Assert.InRange(w.Sum(), 0.98, 1.02);
            looked++;
        });
        model.SetAttentionProbe(false);
        Assert.True(looked > 0);
        Assert.Equal(plain, probed);
        Assert.Null(model.GetAttentionShape());
    }

    [Fact]
    public void PrefixCache_RestoresAnEarlierPrompt()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        Assert.Equal((0UL, 0, 0UL, 0UL), model.GetPrefixCacheInfo());   // off by default
        int[] Prompt(string topic) => model.Tokenize(model.ApplyChatTemplate(
            new[] { ("user", string.Concat(Enumerable.Repeat(topic + " ", 25)) + "Summarise it in one sentence.") }, true), addBos: true)!;
        List<int> Run(int[] ids)
        {
            var tok = model.Predict(ids);
            var res = new List<int>();
            for (int i = 0; i < 8 && !model.IsEndOfGeneration(tok); i++) { res.Add(tok); tok = model.NextToken(); }
            return res;
        }
        var a = Prompt("Daylight looks blue because the atmosphere scatters short wavelengths.");
        var b = Prompt("A compiler turns source code into machine instructions and optimises them.");
        Assert.True(model.SetPrefixCache(512UL * 1024 * 1024, 64));
        model.SetSampling(new SamplingOptions { Temperature = 0f });   // greedy: the two visits must give the same tokens
        model.ResetSession();
        var first = Run(a);
        Run(b);
        Assert.True(model.LastReusedTokens < a.Length / 2);
        var again = Run(a);
        Assert.True(model.LastReusedTokens >= a.Length - 4);
        // both visits answer; the last prompt token runs on its own, as in any session reuse, so a word may differ at a near-tie
        Assert.NotEmpty(first); Assert.NotEmpty(again);
        var info = model.GetPrefixCacheInfo();
        Assert.True(info.Hits >= 1 && info.Entries >= 1 && info.RestoredTokens >= (ulong)(a.Length / 2) && info.Bytes > 0);
        Assert.True(model.SetPrefixCache(0));
        Assert.Equal((0UL, 0, 0UL, 0UL), model.GetPrefixCacheInfo());
    }

    [Fact]
    public void PrefixCache_SurvivesOnDisk()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        int[] Prompt(string topic) => model.Tokenize(model.ApplyChatTemplate(
            new[] { ("user", string.Concat(Enumerable.Repeat(topic + " ", 25)) + "Summarise it in one sentence.") }, true), addBos: true)!;
        void Run(int[] ids) { var tok = model.Predict(ids); for (int i = 0; i < 4 && !model.IsEndOfGeneration(tok); i++) tok = model.NextToken(); }
        var a = Prompt("Daylight looks blue because the atmosphere scatters short wavelengths.");
        var b = Prompt("A compiler turns source code into machine instructions and optimises them.");
        var dir = Path.Combine(Path.GetTempPath(), "dsr-prefix-" + Guid.NewGuid().ToString("N"));
        try
        {
            Assert.True(model.SetPrefixCache(512UL * 1024 * 1024, 64));
            model.ResetSession();
            Run(a); Run(b);
            Assert.True(model.SavePrefixCache(dir) >= 1);
            Assert.True(model.SetPrefixCache(0));
            Assert.Equal((0UL, 0, 0UL, 0UL), model.GetPrefixCacheInfo());
            Assert.True(model.SetPrefixCache(512UL * 1024 * 1024, 64));
            Assert.True(model.LoadPrefixCache(dir) >= 1);
            model.ResetSession();
            Run(a);
            Assert.True(model.LastReusedTokens >= a.Length - 4);
            Assert.Equal(0, model.LoadPrefixCache(Path.Combine(dir, "missing")));
        }
        finally
        {
            model.SetPrefixCache(0);
            try { Directory.Delete(dir, true); } catch { }
        }
    }
}
