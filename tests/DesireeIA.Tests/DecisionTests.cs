// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

// Decision models (non-autoregressive encoder + typed head, e.g. Laya). The
// real-model tests need the model DIRECTORY in DESIREEIA_TEST_DECISION_MODEL
// and are skipped without it, like the other real-model tests.
[Collection("native-model")]
public class DecisionTests
{
    private static string? ModelDir()
    {
        var dir = Environment.GetEnvironmentVariable("DESIREEIA_TEST_DECISION_MODEL");
        return !string.IsNullOrEmpty(dir) && Directory.Exists(dir) ? dir : null;
    }

    private static bool NativeHasDecide()
    {
        try
        {
            return DesireeIAEngine.NativeAbiVersion >= 6;
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

    [Fact]
    public void Questions_AreBuiltFromTheFactories()
    {
        var choice = DecisionQuestion.Choice("route it", new Dictionary<string, string> { ["a"] = "x", ["b"] = "" });
        Assert.Equal(DecisionQuestionType.Choice, choice.Type);
        Assert.Equal(2, choice.Options.Count);

        var labels = DecisionQuestion.Choice("pick", "yes", "no");
        Assert.Equal("yes", labels.Options[0].Key);
        Assert.Equal(string.Empty, labels.Options[0].Value);

        var score = DecisionQuestion.Score("rate", new[] { "bad", "ok", "good" });
        Assert.Equal(3, score.Levels.Count);

        var noul = DecisionQuestion.NoUl("holds?", trueDescription: "it does");
        Assert.Equal("it does", noul.TrueDescription);
        Assert.Null(noul.FalseDescription);
    }

    [Fact]
    public void Decide_PicksTheRightDepartment_OnRealModel()
    {
        if (!NativeHasDecide()) return;
        var dir = ModelDir();
        if (dir is null) return;

        using var model = LocalModel.Load(dir, new ExecutionPlan());
        var result = model.Decide(
            "Subject: Server down in production\n\nThe main API has been returning 500 errors for 20 minutes.",
            new[]
            {
                DecisionQuestion.Choice("Route this email to the right team.", new Dictionary<string, string>
                {
                    ["billing"] = "payment and invoice questions",
                    ["infra"] = "servers, outages and deployments",
                    ["sales"] = "new customers and pricing"
                })
            });

        Assert.NotNull(result);
        var answer = Assert.Single(result!.Answers);
        Assert.Equal(new[] { "billing", "infra", "sales" }, answer.Options);
        Assert.Equal("infra", answer.Choice);
        Assert.Equal(1, answer.Index);
        Assert.InRange(answer.Probabilities.Sum(), 0.999, 1.001);
        Assert.InRange(answer.Confidence, 0.0, 1.0);
        Assert.True(answer.TokenCount > 0);
    }

    [Fact]
    public void Decide_AnswersEveryQuestionType_OnRealModel()
    {
        if (!NativeHasDecide()) return;
        var dir = ModelDir();
        if (dir is null) return;

        using var model = LocalModel.Load(dir, new ExecutionPlan());
        var result = model.Decide(
            new { ticket = "I was charged twice for my subscription.", tier = "gold" },
            new[]
            {
                DecisionQuestion.Choice("Which department handles this?", "billing", "support", "engineering"),
                DecisionQuestion.Score("Rate the urgency.", new[] { "none", "low", "medium", "high", "critical" }),
                DecisionQuestion.NoUl("The customer asks for a refund.")
            });

        Assert.NotNull(result);
        Assert.Equal(3, result!.Answers.Count);
        Assert.Equal("billing", result.Answers[0].Choice);
        Assert.Equal(5, result.Answers[1].Probabilities.Count);
        Assert.Equal(new[] { "false", "true" }, result.Answers[2].Options);
    }

    [Fact]
    public void Decide_RejectsAnInvalidRequest_OnRealModel()
    {
        if (!NativeHasDecide()) return;
        var dir = ModelDir();
        if (dir is null) return;

        using var model = LocalModel.Load(dir, new ExecutionPlan());
        // A choice with a single option cannot be decided.
        Assert.Throws<InvalidOperationException>(() =>
            model.Decide("text", new[] { DecisionQuestion.Choice("only one", "a") }));
    }
}
