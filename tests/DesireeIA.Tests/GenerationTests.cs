// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

public class StopSequenceScannerTests
{
    [Fact]
    public void NoStops_EmitsEverythingImmediately()
    {
        var scanner = new StopSequenceScanner(null);
        var (emit, stopped) = scanner.Feed("hello world");
        Assert.Equal("hello world", emit);
        Assert.False(stopped);
    }

    [Fact]
    public void StopSequence_WithinSinglePiece_TruncatesBeforeIt()
    {
        var scanner = new StopSequenceScanner(new[] { "STOP" });
        var (emit, stopped) = scanner.Feed("hello STOP world");
        Assert.Equal("hello ", emit);
        Assert.True(stopped);
    }

    [Fact]
    public void StopSequence_SplitAcrossPieces_IsStillDetected()
    {
        var scanner = new StopSequenceScanner(new[] { "</s>" });
        var (emit1, stopped1) = scanner.Feed("hi the");
        Assert.False(stopped1);
        var (emit2, stopped2) = scanner.Feed("re</");
        Assert.False(stopped2);
        var (emit3, stopped3) = scanner.Feed("s>bye");
        Assert.True(stopped3);
        Assert.Equal("hi there", emit1 + emit2 + emit3);
    }

    [Fact]
    public void Flush_ReturnsBufferedTailWhenNoStopEverMatched()
    {
        var scanner = new StopSequenceScanner(new[] { "ZZZZZ" });
        var (emit, stopped) = scanner.Feed("abc");
        Assert.False(stopped);
        var tail = scanner.Flush();
        Assert.Equal("abc", emit + tail);
    }

    [Fact]
    public void MultipleStopSequences_FirstMatchWins()
    {
        var scanner = new StopSequenceScanner(new[] { "\n\n", "END" });
        var (emit, stopped) = scanner.Feed("line1\n\nEND");
        Assert.Equal("line1", emit);
        Assert.True(stopped);
    }
}

public class ToolCallingTests
{
    private static readonly ToolDefinition[] Tools =
    {
        new("get_weather", "Returns current weather for a city", "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}")
    };

    [Fact]
    public void BuildSystemPrompt_ListsToolNameAndSchema()
    {
        var prompt = ToolCalling.BuildSystemPrompt(Tools);
        Assert.Contains("get_weather", prompt);
        Assert.Contains("city", prompt);
        Assert.Contains("<tool_call>", prompt);
    }

    [Fact]
    public void TryParse_ValidToolCallBlock_Succeeds()
    {
        var response = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Roma\"}}\n</tool_call>";
        var ok = ToolCalling.TryParse(response, out var call);
        Assert.True(ok);
        Assert.Equal("get_weather", call!.Name);
        Assert.Contains("Roma", call.ArgumentsJson);
    }

    [Fact]
    public void TryParse_NoToolCallBlock_Fails()
    {
        var ok = ToolCalling.TryParse("just a normal answer", out var call);
        Assert.False(ok);
        Assert.Null(call);
    }

    [Fact]
    public void TryParse_MalformedJson_Fails()
    {
        var response = "<tool_call>not json</tool_call>";
        var ok = ToolCalling.TryParse(response, out var call);
        Assert.False(ok);
        Assert.Null(call);
    }

    [Fact]
    public void TryParse_MissingName_Fails()
    {
        var response = "<tool_call>{\"arguments\": {}}</tool_call>";
        var ok = ToolCalling.TryParse(response, out var call);
        Assert.False(ok);
    }

    [Fact]
    public void BuildResultMessage_UsesUserRoleForTemplateCompatibility()
    {
        var (role, content) = ToolCalling.BuildResultMessage("get_weather", "{\"tempC\": 21}");
        Assert.Equal("user", role);
        Assert.Contains("get_weather", content);
        Assert.Contains("21", content);
    }
}

public class StructuredOutputTests
{
    [Fact]
    public void BuildJsonInstruction_WithoutSchema_MentionsJsonOnly()
    {
        var instr = StructuredOutput.BuildJsonInstruction();
        Assert.Contains("JSON", instr);
    }

    [Fact]
    public void BuildJsonInstruction_WithSchema_IncludesSchema()
    {
        var instr = StructuredOutput.BuildJsonInstruction("{\"type\":\"object\"}");
        Assert.Contains("{\"type\":\"object\"}", instr);
    }

    [Fact]
    public void TryExtractJson_PlainObject_ReturnsIt()
    {
        var json = StructuredOutput.TryExtractJson("{\"a\": 1}");
        Assert.Equal("{\"a\": 1}", json);
    }

    [Fact]
    public void TryExtractJson_ObjectWrappedInProseAndFences_ExtractsInnerObject()
    {
        var text = "Sure, here you go:\n```json\n{\"a\": 1, \"b\": [1,2,3]}\n```\nHope that helps!";
        var json = StructuredOutput.TryExtractJson(text);
        Assert.NotNull(json);
        Assert.Equal("{\"a\": 1, \"b\": [1,2,3]}", json);
    }

    [Fact]
    public void TryExtractJson_ArrayTopLevel_ReturnsIt()
    {
        var json = StructuredOutput.TryExtractJson("prefix [1, 2, 3] suffix");
        Assert.Equal("[1, 2, 3]", json);
    }

    [Fact]
    public void TryExtractJson_NoJsonPresent_ReturnsNull()
    {
        var json = StructuredOutput.TryExtractJson("no json here at all");
        Assert.Null(json);
    }

    [Fact]
    public void TryExtractJson_StringContainingBracesIsNotConfused()
    {
        var text = "{\"msg\": \"looks like { not json }\"}";
        var json = StructuredOutput.TryExtractJson(text);
        Assert.Equal(text, json);
    }
}
