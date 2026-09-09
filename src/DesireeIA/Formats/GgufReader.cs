using System.Buffers.Binary;
using System.Text;

namespace DesireeIA.Formats;

public sealed class GgufReader
{
    private const uint GgufMagic = 0x46554747;

    private enum ValueType : uint
    {
        UInt8 = 0,
        Int8 = 1,
        UInt16 = 2,
        Int16 = 3,
        UInt32 = 4,
        Int32 = 5,
        Float32 = 6,
        Bool = 7,
        String = 8,
        Array = 9,
        UInt64 = 10,
        Int64 = 11,
        Float64 = 12
    }

    private readonly Dictionary<string, object> _kv = new();

    /// <summary>
    /// Raw metadata key/value pairs as parsed, for diagnostics (e.g. checking
    /// which architecture-specific keys a given GGUF file actually carries).
    /// Array values appear as an <see cref="ArrayMarker"/> placeholder, not
    /// their elements.
    /// </summary>
    public IReadOnlyDictionary<string, object> RawKeyValues => _kv;

    public ModelMetadata Open(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 64 * 1024, options: FileOptions.SequentialScan);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        var header = ReadHeader(reader);
        ReadKeyValues(reader, header.KeyValueCount);
        _tensors = ReadTensorInfos(reader, header.TensorCount);

        var arch = GetString("general.architecture") ?? "unknown";
        var contextLength = GetUInt64($"{arch}.context_length")
                            ?? GetUInt64("gguf.context_length") ?? 0;
        var layerCount = GetUInt64($"{arch}.block_count") ?? 0;
        var embedding = GetUInt64($"{arch}.embedding_length") ?? 0;
        var embeddingType = embedding > 0
            ? DetectTensorType($"{arch}.token_embd.weight")
              ?? DetectTensorType("token_embd.weight")
              ?? TensorType.Unknown
            : TensorType.Unknown;

        uint vocabSize = 0;
        var vocabCount = GetArrayLength("tokenizer.ggml.tokens");
        if (vocabCount is > 0)
        {
            vocabSize = (uint)vocabCount.Value;
        }
        else
        {
            vocabSize = (uint)(GetUInt64("general.vocab_size") ?? 0);
        }

        return new ModelMetadata(
            Version: header.Version,
            TensorCount: header.TensorCount,
            KeyValueCount: header.KeyValueCount,
            Architecture: arch,
            ContextLength: (uint)contextLength,
            VocabularySize: vocabSize,
            LayerCount: (uint)layerCount,
            EmbeddingSize: (uint)embedding,
            EmbeddingType: embeddingType,
            Tensors: _tensors);
    }

    private static (uint Version, ulong TensorCount, ulong KeyValueCount) ReadHeader(BinaryReader r)
    {
        Span<byte> magic = stackalloc byte[4];
        r.Read(magic);
        if (!magic.SequenceEqual("GGUF"u8))
        {
            throw new InvalidDataException("Not a GGUF file.");
        }
        uint version = r.ReadUInt32();
        ulong tensorCount = r.ReadUInt64();
        ulong kvCount = r.ReadUInt64();
        return (version, tensorCount, kvCount);
    }

    private void ReadKeyValues(BinaryReader r, ulong count)
    {
        for (ulong i = 0; i < count; i++)
        {
            var key = ReadString(r);
            var type = (ValueType)r.ReadUInt32();
            object? value = ReadValue(r, type, out var arrayLength);
            _kv[key] = arrayLength is null ? value! : new ArrayMarker(arrayLength.Value);
        }
    }

    private static List<TensorEntry> ReadTensorInfos(BinaryReader r, ulong count)
    {
        var list = new List<TensorEntry>((int)Math.Min(count, 100_000));
        for (ulong i = 0; i < count; i++)
        {
            var name = ReadString(r);
            uint nDims = r.ReadUInt32();
            var dims = new long[nDims];
            for (uint d = 0; d < nDims; d++)
            {
                ulong dim = r.ReadUInt64();
                dims[(int)d] = (long)dim;
            }
            var type = (TensorType)r.ReadUInt32();
            ulong offset = r.ReadUInt64();
            list.Add(new TensorEntry(name, type, dims, offset));
        }
        return list;
    }

    private object? ReadValue(BinaryReader r, ValueType type, out long? arrayLength)
    {
        arrayLength = null;
        switch (type)
        {
            case ValueType.UInt8: return r.ReadByte();
            case ValueType.Int8: return r.ReadSByte();
            case ValueType.UInt16: return r.ReadUInt16();
            case ValueType.Int16: return r.ReadInt16();
            case ValueType.UInt32: return r.ReadUInt32();
            case ValueType.Int32: return r.ReadInt32();
            case ValueType.Float32: return r.ReadSingle();
            case ValueType.Bool: return r.ReadBoolean();
            case ValueType.String: return ReadString(r);
            case ValueType.UInt64: return r.ReadUInt64();
            case ValueType.Int64: return r.ReadInt64();
            case ValueType.Float64: return r.ReadDouble();
            case ValueType.Array:
            {
                var elemType = (ValueType)r.ReadUInt32();
                ulong n = r.ReadUInt64();
                arrayLength = (long)n;
                for (ulong i = 0; i < n; i++)
                {
                    ReadValue(r, elemType, out _);
                }
                return 0;
            }
            default:
                throw new InvalidDataException($"Unknown GGUF value type {type}.");
        }
    }

    private static string ReadString(BinaryReader r)
    {
        ulong len = r.ReadUInt64();
        var bytes = r.ReadBytes((int)len);
        return Encoding.UTF8.GetString(bytes);
    }

    private string? GetString(string key) => _kv.TryGetValue(key, out var v) ? v as string : null;

    private ulong? GetUInt64(string key)
    {
        if (!_kv.TryGetValue(key, out var v)) return null;
        return v switch
        {
            ulong u => u,
            long l => (ulong)Math.Max(l, 0),
            uint ui => ui,
            int i => (ulong)Math.Max(i, 0),
            _ => null
        };
    }

    private long? GetArrayLength(string key) =>
        _kv.TryGetValue(key, out var v) && v is ArrayMarker m ? m.Length : null;

    private TensorType? DetectTensorType(string tensorName)
    {
        foreach (var t in _tensors)
        {
            if (string.Equals(t.Name, tensorName, StringComparison.Ordinal))
            {
                return t.Type;
            }
        }
        return null;
    }

    private List<TensorEntry> _tensors = new();

    private sealed record ArrayMarker(long Length);
}