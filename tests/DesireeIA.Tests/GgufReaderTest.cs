// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Text;
using DesireeIA.Formats;
using Xunit;

namespace DesireeIA.Tests;

public class GgufReaderTest
{
    private static byte[] BuildSyntheticGguf()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write("GGUF"u8);
        w.Write(3u);
        w.Write(3ul);
        w.Write(5ul);

        void WriteString(string s)
        {
            var b = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)b.Length);
            w.Write(b);
        }

        void WriteKv(string key, Action writeValue)
        {
            WriteString(key);
            writeValue();
        }

        WriteKv("general.architecture", () => { w.Write(8u); WriteString("gemma3"); });
        WriteKv("gemma3.context_length", () => { w.Write(10u); w.Write(8192ul); });
        WriteKv("gemma3.block_count", () => { w.Write(10u); w.Write(4ul); });
        WriteKv("gemma3.embedding_length", () => { w.Write(10u); w.Write(128ul); });

        WriteString("tokenizer.ggml.tokens");
        w.Write(9u);
        w.Write(8u);
        w.Write(100ul);
        for (int i = 0; i < 100; i++) WriteString($"tok{i}");

        void WriteTensor(string name, uint type, ulong[] dims)
        {
            WriteString(name);
            w.Write((uint)dims.Length);
            foreach (var d in dims) w.Write(d);
            w.Write(type);
            w.Write(0ul);
        }

        WriteTensor("token_embd.weight", 14u, new ulong[] { 128, 100 });
        WriteTensor("blk.0.ffn_gate.weight", 12u, new ulong[] { 128, 512 });
        WriteTensor("output_norm.weight", 0u, new ulong[] { 128 });

        w.Flush();
        return ms.ToArray();
    }

    [Fact]
    public void ParsesSyntheticGguf()
    {
        var path = Path.Combine(Path.GetTempPath(), $"desireeia_test_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, BuildSyntheticGguf());
        try
        {
            var reader = new GgufReader();
            var meta = reader.Open(path);

            Assert.Equal(3u, meta.Version);
            Assert.Equal("gemma3", meta.Architecture);
            Assert.Equal(8192u, meta.ContextLength);
            Assert.Equal(4u, meta.LayerCount);
            Assert.Equal(128u, meta.EmbeddingSize);
            Assert.Equal(100u, meta.VocabularySize);
            Assert.Equal(3ul, meta.TensorCount);
            Assert.Equal(TensorType.Q6K, meta.EmbeddingType);

            var gate = Assert.Single(meta.Tensors, t => t.Name == "blk.0.ffn_gate.weight");
            Assert.Equal(TensorType.Q4K, gate.Type);
            Assert.Equal(new long[] { 128, 512 }, gate.Shape);
        }
        finally
        {
            File.Delete(path);
        }
    }
}