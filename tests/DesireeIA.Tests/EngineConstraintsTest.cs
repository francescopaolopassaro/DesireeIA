// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Text;
using System.Text.Json;
using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

/// <summary>JSON schema constraint, scoring and logit bias (engine 0.3.0). Skipped without a model or an engine that has them.</summary>
[Collection("native-model")]
public class EngineConstraintsTest
{
    private const string Schema =
        """{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"},"kind":{"enum":["village","town","city"]}},"required":["city","population","kind"]}""";

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

    private static int[] Prompt(LocalModel model, string user)
    {
        var text = model.ApplyChatTemplate(new[] { ("user", user) }, true);
        return model.Tokenize(text, addBos: true)!;
    }

    private static string Generate(LocalModel model, string user, int limit)
    {
        var ids = Prompt(model, user);
        model.ResetSession();
        var tok = model.Predict(ids);
        var sb = new StringBuilder();
        for (int i = 0; i < limit && !model.IsEndOfGeneration(tok); i++)
        {
            sb.Append(model.TokenPiece(tok));
            tok = model.NextToken();
        }
        return sb.ToString();
    }

    [Fact]
    public void JsonSchema_MakesEveryAnswerParseWithTheRequiredKeys()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        try
        {
            model.SetJsonSchema(Schema);
        }
        catch (NotSupportedException) { return; }
        foreach (var prompt in new[] { "Describe Rome.", "Tell me about a tiny village.", "Say hello." })
        {
            using var doc = JsonDocument.Parse(Generate(model, prompt, 160));
            var root = doc.RootElement;
            Assert.Equal(JsonValueKind.String, root.GetProperty("city").ValueKind);
            Assert.True(root.GetProperty("population").TryGetInt64(out _));
            Assert.Contains(root.GetProperty("kind").GetString(), new[] { "village", "town", "city" });
        }
        model.SetJsonSchema(null);
    }

    [Fact]
    public void JsonSchema_RefusesASchemaItCannotEnforce()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        try
        {
            Assert.Throws<ArgumentException>(() => model.SetJsonSchema("""{"type":"bogus"}"""));
            Assert.Throws<ArgumentException>(() => model.SetJsonSchema("{not json"));
        }
        catch (NotSupportedException) { }
    }

    [Fact]
    public void ScoreTokens_GivesProbabilitiesAndEntropies_AndAKnownSentenceIsLessSurprising()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        var good = model.Tokenize("The quick brown fox jumps over the lazy dog.", addBos: true)!;
        var shuffled = model.Tokenize("dog the over lazy jumps fox brown quick The.", addBos: true)!;
        (float[] LogProb, float[] Entropy) a, b;
        try
        {
            a = model.ScoreTokens(good, 1);
            b = model.ScoreTokens(shuffled, 1);
        }
        catch (NotSupportedException) { return; }
        Assert.Equal(good.Length - 1, a.LogProb.Length);
        Assert.All(a.LogProb, v => Assert.True(v <= 1e-4f));
        Assert.All(a.Entropy, v => Assert.True(v >= -1e-4f));
        Assert.True(a.LogProb.Sum() > b.LogProb.Sum());
    }

    [Fact]
    public void LogitBias_ForbidsAndForcesAToken()
    {
        if (!Ready(out var path)) return;
        using var model = LocalModel.Load(path, DesireeIAEngine.BuildPlan(path));
        var ids = Prompt(model, "Say hi.");
        model.ResetSession();
        var first = model.Predict(ids);
        try
        {
            model.SetLogitBias(new Dictionary<int, float> { [first] = -100f });
            model.ResetSession();
            Assert.NotEqual(first, model.Predict(ids));
            var other = (first + 17) % 1000;
            model.SetLogitBias(new Dictionary<int, float> { [other] = 100f });
            model.ResetSession();
            Assert.Equal(other, model.Predict(ids));
        }
        catch (NotSupportedException) { return; }
        finally
        {
            try { model.SetLogitBias(null); } catch (NotSupportedException) { }
        }
        model.ResetSession();
        Assert.Equal(first, model.Predict(ids));
    }
}
