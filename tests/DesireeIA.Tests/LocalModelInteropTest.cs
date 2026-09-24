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

    private static async Task<string> Collect(IAsyncEnumerable<string> stream)
    {
        var sb = new System.Text.StringBuilder();
        await foreach (var piece in stream) sb.Append(piece);
        return sb.ToString();
    }
}
