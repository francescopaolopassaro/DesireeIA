namespace DesireeIA.Formats;

public enum TensorType
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    Bf16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    Unknown = -1
}

public sealed record TensorEntry(string Name, TensorType Type, long[] Shape, ulong Offset);

public sealed record ModelMetadata(
    uint Version,
    ulong TensorCount,
    ulong KeyValueCount,
    string Architecture,
    uint ContextLength,
    uint VocabularySize,
    uint LayerCount,
    uint EmbeddingSize,
    TensorType EmbeddingType,
    IReadOnlyList<TensorEntry> Tensors);