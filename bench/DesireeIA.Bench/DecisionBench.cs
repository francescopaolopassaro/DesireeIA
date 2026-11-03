// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Diagnostics;
using System.Text.Json;
using DesireeIA;

/// <summary>
/// Decision models (Laya): one example of each question type, then the latency of the same question over inputs of
/// growing length.
///
///   DesireeIA.Bench decide path\to\laya\typed-decisions [--lengths 64,128,256,512,1024] [--runs 5] [--json out.json]
/// </summary>
internal static class DecisionBench
{
    private const string Sentence =
        "The quarterly report discusses market volatility, bond yields and equity valuations in detail. ";

    private static readonly DecisionQuestion Topic = DecisionQuestion.Choice(
        "What is the main topic of the text?", new Dictionary<string, string>
        {
            ["cooking"] = "recipes and food",
            ["finance"] = "money and markets",
            ["sports"] = "games and athletes",
            ["travel"] = "trips and places"
        });

    public static int Run(string[] args)
    {
        string? dir = null, jsonOut = null;
        var lengths = new[] { 64, 128, 256, 512, 1024 };
        int runs = 5;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--lengths": lengths = args[++i].Split(',').Select(int.Parse).ToArray(); break;
                case "--runs": runs = int.Parse(args[++i]); break;
                case "--json": jsonOut = args[++i]; break;
                default: dir = args[i]; break;
            }
        }
        if (dir is null || !Directory.Exists(dir))
        {
            Console.Error.WriteLine("usage: DesireeIA.Bench decide <model-folder> [--lengths 64,256,1024] [--runs 5] [--json out.json]");
            return 1;
        }

        var sw = Stopwatch.StartNew();
        using var model = LocalModel.Load(dir, new ExecutionPlan());
        var loadSeconds = sw.Elapsed.TotalSeconds;
        Console.WriteLine($"DesireeIA native engine interface {DesireeIAEngine.NativeAbiVersion}, loaded {dir} in {loadSeconds:F2} s\n");

        Demo(model);

        Console.WriteLine($"\n{"input tokens",13} {"median ms",10} {"min ms",8} {"tokens/s",9}");
        var rows = new List<object>();
        foreach (var target in lengths)
        {
            int reps = RepsFor(model, target);
            Once(model, reps);                                   // warm-up
            var times = new List<double>();
            int tokens = 0;
            for (int r = 0; r < runs; r++)
            {
                var (seconds, n) = Once(model, reps);
                times.Add(seconds);
                tokens = n;
            }
            times.Sort();
            double median = times[times.Count / 2];
            Console.WriteLine($"{tokens,13} {median * 1000,10:F0} {times[0] * 1000,8:F0} {tokens / median,9:F0}");
            rows.Add(new { input_tokens = tokens, median_ms = median * 1000, min_ms = times[0] * 1000, tokens_per_second = tokens / median, runs });
        }

        if (jsonOut is not null)
        {
            File.WriteAllText(jsonOut, JsonSerializer.Serialize(
                new { load_seconds = loadSeconds, results = rows }, new JsonSerializerOptions { WriteIndented = true }));
        }
        return 0;
    }

    private static void Demo(LocalModel model)
    {
        var result = model.Decide(
            "Subject: Server down in production\n\nHi team, the main API has been returning 500 errors for the last 20 minutes and customers are complaining.",
            new[]
            {
                DecisionQuestion.Choice("Route this email to the right team.", new Dictionary<string, string>
                {
                    ["billing"] = "payment and invoice questions",
                    ["infra"] = "servers, outages and deployments",
                    ["sales"] = "new customers and pricing"
                }),
                DecisionQuestion.Score("Rate the urgency.", new[] { "none", "low", "medium", "high", "critical" }),
                DecisionQuestion.NoUl("Customers are affected.")
            });
        if (result is null)
        {
            Console.Error.WriteLine("This model is not a decision model.");
            Environment.Exit(1);
        }
        foreach (var a in result!.Answers)
        {
            var probs = string.Join(", ", a.Options.Zip(a.Probabilities, (o, p) => $"{o}={p:F2}"));
            Console.WriteLine($"{a.Options.Count,2} options -> {a.Choice,-8} confidence {a.Confidence:F2}   [{probs}]");
        }
    }

    private static int _salt;

    private static (double Seconds, int Tokens) Once(LocalModel model, int reps)
    {
        // A different state every call: the engine answers an identical request that repeats back to back from
        // memory, which would time as 0 ms.
        var state = string.Concat(Enumerable.Repeat(Sentence, reps)) + $"[{++_salt}]";
        var sw = Stopwatch.StartNew();
        var result = model.Decide(state, new[] { Topic });
        return (sw.Elapsed.TotalSeconds, result!.Answers[0].TokenCount);
    }

    // Repetitions of the sentence that make the whole input about `target` tokens.
    private static int RepsFor(LocalModel model, int target)
    {
        var (_, small) = Once(model, 1);
        var (_, large) = Once(model, 11);
        double per = Math.Max(1.0, (large - small) / 10.0);
        double overhead = small - per;
        return Math.Max(1, (int) Math.Round((target - overhead) / per));
    }
}
