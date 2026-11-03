// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace DesireeIA;

/// <summary>Kind of a question put to a decision model.</summary>
public enum DecisionQuestionType
{
    /// <summary>One of N labelled options.</summary>
    Choice,
    /// <summary>An ordinal level 0..N-1.</summary>
    Score,
    /// <summary>A yes/no statement.</summary>
    NoUl
}

/// <summary>
/// A typed question for a decision model (for example Laya): the model never
/// generates text, it picks among the options the caller defines here.
/// Create one with <see cref="Choice"/>, <see cref="Score"/> or <see cref="NoUl"/>.
/// </summary>
public sealed class DecisionQuestion
{
    public DecisionQuestionType Type { get; }
    public string Instruction { get; }

    /// <summary>Choice: label and description pairs, in option order.</summary>
    public IReadOnlyList<KeyValuePair<string, string>> Options { get; }

    /// <summary>Score: the text describing each level, level 0 first.</summary>
    public IReadOnlyList<string> Levels { get; }

    /// <summary>NoUl: optional text for the "true" and "false" outcomes.</summary>
    public string? TrueDescription { get; }
    public string? FalseDescription { get; }

    private DecisionQuestion(DecisionQuestionType type, string instruction,
        IReadOnlyList<KeyValuePair<string, string>>? options, IReadOnlyList<string>? levels,
        string? trueDescription, string? falseDescription)
    {
        Type = type;
        Instruction = instruction;
        Options = options ?? Array.Empty<KeyValuePair<string, string>>();
        Levels = levels ?? Array.Empty<string>();
        TrueDescription = trueDescription;
        FalseDescription = falseDescription;
    }

    /// <summary>Pick one of the labelled options. An empty description is allowed.</summary>
    public static DecisionQuestion Choice(string instruction, IEnumerable<KeyValuePair<string, string>> options)
        => new(DecisionQuestionType.Choice, instruction, options.ToList(), null, null, null);

    /// <summary>Pick one of the labels; the label itself is the option text.</summary>
    public static DecisionQuestion Choice(string instruction, params string[] labels)
        => Choice(instruction, labels.Select(l => new KeyValuePair<string, string>(l, string.Empty)));

    /// <summary>Rate on an ordinal scale; <paramref name="levels"/>[i] describes level i.</summary>
    public static DecisionQuestion Score(string instruction, IEnumerable<string> levels)
        => new(DecisionQuestionType.Score, instruction, null, levels.ToList(), null, null);

    /// <summary>Decide whether a statement holds.</summary>
    public static DecisionQuestion NoUl(string instruction, string? trueDescription = null, string? falseDescription = null)
        => new(DecisionQuestionType.NoUl, instruction, null, null, trueDescription, falseDescription);
}

/// <summary>The model's answer to one <see cref="DecisionQuestion"/>.</summary>
public sealed class DecisionAnswer
{
    /// <summary>Option labels, in the order of the question ("false","true" for NoUl, "0".."N-1" for Score).</summary>
    public IReadOnlyList<string> Options { get; init; } = Array.Empty<string>();
    /// <summary>Probability of each option (calibrated unless calibration was turned off).</summary>
    public IReadOnlyList<float> Probabilities { get; init; } = Array.Empty<float>();
    /// <summary>Raw scores before calibration and softmax.</summary>
    public IReadOnlyList<float> Logits { get; init; } = Array.Empty<float>();
    /// <summary>Index of the most probable option.</summary>
    public int Index { get; init; }
    /// <summary>Label of the most probable option.</summary>
    public string Choice { get; init; } = string.Empty;
    /// <summary>1 minus the normalized entropy of <see cref="Probabilities"/>: 0 = undecided, 1 = certain.</summary>
    public double Confidence { get; init; }
    /// <summary>First output of the act/escalate head. The model's authors report it as unreliable: prefer <see cref="Confidence"/>.</summary>
    public double ActProbability { get; init; }
    public IReadOnlyList<float> ActProbabilities { get; init; } = Array.Empty<float>();
    /// <summary>Input length the question needed, in tokens.</summary>
    public int TokenCount { get; init; }
}

/// <summary>Answers of a decision model, one per question and in the same order.</summary>
public sealed class DecisionResult
{
    public string Model { get; init; } = string.Empty;
    public IReadOnlyList<DecisionAnswer> Answers { get; init; } = Array.Empty<DecisionAnswer>();
}

internal static class DecisionJson
{
    private static readonly JsonWriterOptions WriterOptions = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        Indented = false
    };

    public static string BuildRequest(object? state, IReadOnlyList<DecisionQuestion> questions,
                                      bool calibrate, bool truncateLeft)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, WriterOptions))
        {
            w.WriteStartObject();
            w.WritePropertyName("state");
            if (state is string s) w.WriteStringValue(s);
            else if (state is JsonElement je) je.WriteTo(w);
            else JsonSerializer.SerializeToElement(state).WriteTo(w);

            w.WriteStartArray("questions");
            foreach (var q in questions)
            {
                w.WriteStartObject();
                w.WriteString("type", q.Type switch
                {
                    DecisionQuestionType.Choice => "choice",
                    DecisionQuestionType.Score => "score",
                    _ => "noul"
                });
                w.WriteString("instruction", q.Instruction);
                switch (q.Type)
                {
                    case DecisionQuestionType.Choice:
                        w.WriteStartObject("options");
                        foreach (var kv in q.Options) w.WriteString(kv.Key, kv.Value);
                        w.WriteEndObject();
                        break;
                    case DecisionQuestionType.Score:
                        w.WriteStartArray("options");
                        foreach (var l in q.Levels) w.WriteStringValue(l);
                        w.WriteEndArray();
                        break;
                    default:
                        if (!string.IsNullOrEmpty(q.TrueDescription)) w.WriteString("true", q.TrueDescription);
                        if (!string.IsNullOrEmpty(q.FalseDescription)) w.WriteString("false", q.FalseDescription);
                        break;
                }
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteBoolean("calibrate", calibrate);
            w.WriteBoolean("truncate_left", truncateLeft);
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(ms.ToArray());
    }

    public static DecisionResult ParseResponse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var answers = new List<DecisionAnswer>();
        foreach (var a in root.GetProperty("answers").EnumerateArray())
        {
            answers.Add(new DecisionAnswer
            {
                Options = a.GetProperty("options").EnumerateArray().Select(e => e.GetString() ?? string.Empty).ToArray(),
                Probabilities = Floats(a.GetProperty("probabilities")),
                Logits = Floats(a.GetProperty("logits")),
                Index = a.GetProperty("index").GetInt32(),
                Choice = a.GetProperty("choice").GetString() ?? string.Empty,
                Confidence = a.GetProperty("confidence").GetDouble(),
                ActProbability = a.GetProperty("act_probability").GetDouble(),
                ActProbabilities = Floats(a.GetProperty("act_probabilities")),
                TokenCount = a.GetProperty("n_tokens").GetInt32()
            });
        }
        return new DecisionResult
        {
            Model = root.TryGetProperty("model", out var m) ? m.GetString() ?? string.Empty : string.Empty,
            Answers = answers
        };

        static float[] Floats(JsonElement e) => e.EnumerateArray().Select(x => x.GetSingle()).ToArray();
    }
}
