// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Text;
using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

/// <summary>An emoji split over byte-fallback tokens must come out whole, not as U+FFFD.</summary>
public class Utf8StreamTests
{
    [Fact]
    public void EmojiSplitAcrossTokens_IsEmittedWhole()
    {
        var bytes = Encoding.UTF8.GetBytes("Ciao! \U0001F60A");
        var s = new Utf8Stream();
        var sb = new StringBuilder();
        sb.Append(s.Feed(bytes[..6]));          // "Ciao! "
        foreach (var b in bytes[6..])           // one byte per token, like <0xF0> <0x9F> <0x98> <0x8A>
            sb.Append(s.Feed(new[] { b }));
        sb.Append(s.Flush());

        Assert.Equal("Ciao! \U0001F60A", sb.ToString());
        Assert.DoesNotContain('�', sb.ToString());
    }

    [Fact]
    public void IncompleteSequence_EmitsNothingUntilComplete()
    {
        var e = Encoding.UTF8.GetBytes("è");   // two bytes
        var s = new Utf8Stream();
        Assert.Equal("", s.Feed(new[] { e[0] }));
        Assert.Equal("è", s.Feed(new[] { e[1] }));
    }

    [Fact]
    public void TruncatedAtEnd_FlushGivesOneReplacement()
    {
        var s = new Utf8Stream();
        Assert.Equal("", s.Feed(new byte[] { 0xF0, 0x9F }));
        Assert.Equal("�", s.Flush());
    }
}
