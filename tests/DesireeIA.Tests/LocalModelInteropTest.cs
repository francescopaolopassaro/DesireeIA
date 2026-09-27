// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

public class LocalModelInteropTest
{
    private static bool NativeAvailable()
    {
        try
        {
            _ = DesireeIAEngine.DetectHardware();
            return true;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private static string? ModelPath() =>
        Environment.GetEnvironmentVariable("DESIREEIA_TEST_MODEL_PATH");

    [Fact]
    public void Predict_And_NextToken_RunOnRealModel()
    {
        if (!NativeAvailable()) return;
        var path = ModelPath();
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        var plan = DesireeIAEngine.BuildPlan(path);
        using var model = LocalModel.Load(path, plan);

        var first = model.Predict(new[] { 2 }); // BOS
        Assert.InRange(first, 0, int.MaxValue);

        var second = model.NextToken();
        Assert.InRange(second, 0, int.MaxValue);

        Assert.True(model.ContextSize() > 0);
    }

    // Conversation session: turn 2 re-sends turn 1 + a follow-up. Exact mode
    // reuses the turn-1 prompt from the cache and must answer exactly like a
    // full prefill (greedy sampling is the default, so it's deterministic).
    [Fact]
    public async Task Session_ReusesPrefix_AndMatchesFullPrefill()
    {
        if (!NativeAvailable()) return;
        var path = ModelPath();
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        Assert.Equal(SessionReuseMode.Exact, model.SessionReuse);

        var turn1 = new List<(string, string)> { ("user", "My cat is called Luna. Say hi.") };
        var answer1 = await Collect(model.ChatStreamAsync(turn1, new GenerateOptions { MaxTokens = 16 }));
        Assert.Equal(0, model.LastReusedTokens);

        var turn2 = new List<(string, string)>(turn1) { ("assistant", answer1), ("user", "What is my cat called?") };
        var options = new GenerateOptions { MaxTokens = 16 };
        var withSession = await Collect(model.ChatStreamAsync(turn2, options));
        Assert.True(model.LastReusedTokens > 0, "turn 2 should reuse the turn-1 prompt from the cache");

        model.ResetSession();
        var fromScratch = await Collect(model.ChatStreamAsync(turn2, options));
        Assert.Equal(0, model.LastReusedTokens);
        Assert.Equal(fromScratch, withSession);

        model.SessionReuse = SessionReuseMode.Off;
        await Collect(model.ChatStreamAsync(turn2, options));
        Assert.Equal(0, model.LastReusedTokens);
    }

    // Session file: a prefix prefilled and saved by one model instance is
    // restored by a fresh one (a new process, in practice) and reused - and
    // the answer is the one a full prefill gives. A file for another prefix
    // length or a corrupted one is refused without harm.
    [Fact]
    public void Session_SaveLoad_ReusesPrefixAcrossInstances()
    {
        if (!NativeAvailable()) return;
        var path = ModelPath();
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;
        var file = Path.Combine(Path.GetTempPath(), $"desireeia-session-{Guid.NewGuid():N}.dskv");
        try
        {
            int[] system, turn;
            int[] expected;
            using (var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path)))
            {
                system = model.Tokenize("You are a helpful assistant. Answer briefly and precisely.")!;
                turn = system.Concat(model.Tokenize(" What is two plus two?", addBos: false)!).ToArray();
                expected = Greedy(model, turn, 8);
                model.ResetSession();
                model.Predict(system);
                Assert.True(model.SaveSession(file));
            }
            using (var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path)))
            {
                Assert.Equal(system.Length, model.LoadSession(file));
                var restored = Greedy(model, turn, 8);
                Assert.Equal(system.Length, model.LastReusedTokens);
                Assert.Equal(expected, restored);

                // In memory (what an app that encrypts at rest uses): same result.
                var image = model.SaveSessionBytes(system.Length);
                Assert.NotNull(image);
                model.ResetSession();
                Assert.Equal(system.Length, model.LoadSessionBytes(image!));
                Assert.Equal(expected, Greedy(model, turn, 8));
                Assert.Equal(0, model.LoadSessionBytes(image![..^1]));

                File.WriteAllBytes(file, new byte[] { 1, 2, 3 });
                Assert.Equal(0, model.LoadSession(file));
                Assert.Equal(0, model.LoadSession(file + ".missing"));
            }
        }
        finally
        {
            File.Delete(file);
        }
    }

    [Fact]
    public void Engine_ReportsAbiVersionAndGpus()
    {
        if (!NativeAvailable()) return;
        Assert.True(DesireeIAEngine.NativeAbiVersion >= 2);
        foreach (var g in DesireeIAEngine.Gpus())
        {
            Assert.False(string.IsNullOrWhiteSpace(g.Name));
            Assert.True(g.TotalBytes > 0 && g.FreeBytes <= g.TotalBytes);
        }
    }

    [Fact]
    public void Load_MissingFile_ErrorCarriesTheCause()
    {
        if (!NativeAvailable()) return;
        var ex = Assert.Throws<InvalidOperationException>(() =>
            LocalModel.Load(Path.Combine(Path.GetTempPath(), "does-not-exist.gguf"), DesireeIAEngine.BuildPlan(null)));
        Assert.Contains("-", ex.Message);
    }

    // Cancel from another thread stops a long prefill; the model stays usable
    // and gives the same answer as before.
    [Fact]
    public void Cancel_StopsPrefill_ProgressReportsLayers_TrimKeepsModelUsable()
    {
        if (!NativeAvailable()) return;
        var path = ModelPath();
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        var shortPrompt = model.Tokenize("The capital of France is")!;
        var expected = Greedy(model, shortPrompt, 6);

        long lastDone = -1, total = 0;
        model.PrefillProgress = (d, t) => { lastDone = d; total = t; };
        var filler = model.Tokenize(" the quick brown fox jumps over the lazy dog", addBos: false)!;
        var longPrompt = new List<int>(shortPrompt);
        while (longPrompt.Count < 3000) longPrompt.AddRange(filler);
        model.ResetSession();
        model.Predict(longPrompt.Take(200).ToArray());
        Assert.True(total > 0 && lastDone >= 0, "progress must be reported during a prefill");

        model.ResetSession();
        var worker = Task.Run(() => model.Predict(longPrompt.ToArray()));
        Thread.Sleep(30);
        model.Cancel();
        try { worker.GetAwaiter().GetResult(); } catch (OperationCanceledException) { /* expected when it was still running */ }
        model.PrefillProgress = null;

        Assert.True(model.ReserveContext(1024));
        model.TrimCache();
        model.ResetSession();
        Assert.Equal(expected, Greedy(model, shortPrompt, 6));
    }

    private static int[] Greedy(LocalModel model, int[] prompt, int n)
    {
        var ids = new List<int> { model.Predict(prompt) };
        for (var i = 1; i < n; i++) ids.Add(model.NextToken());
        return ids.ToArray();
    }

    private static async Task<string> Collect(IAsyncEnumerable<string> stream)
    {
        var sb = new System.Text.StringBuilder();
        await foreach (var piece in stream) sb.Append(piece);
        return sb.ToString();
    }
}
