using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DesireeIA.Formats;

if (args.Length < 1)
{
    Console.Error.WriteLine("usage: desireeia_probe <model.gguf>");
    return 1;
}

var path = args[0];
var sw = Stopwatch.StartNew();

Console.WriteLine("== model load ==");
var reader = new GgufReader();
var meta = reader.Open(path);
sw.Stop();
Console.WriteLine($"parse: {sw.ElapsedMilliseconds} ms");
Console.WriteLine($"arch: {meta.Architecture} layers={meta.LayerCount} ctx={meta.ContextLength} " +
                  $"vocab={meta.VocabularySize} tensors={meta.TensorCount}");

var emb = meta.Tensors.FirstOrDefault(t => t.Name.EndsWith("token_embd.weight"));
if (emb is null)
{
    Console.WriteLine("token_embd not found");
    return 1;
}

var sorted = meta.Tensors.OrderBy(t => t.Offset).ToList();
var embIdx = sorted.IndexOf(emb);
ulong end = embIdx + 1 < sorted.Count ? sorted[embIdx + 1].Offset : 0;
ulong bytes = end > emb.Offset ? end - emb.Offset : 0;
Console.WriteLine($"token_embd.weight type={emb.Type} shape={string.Join('x', emb.Shape)} bytes={bytes}");

Console.WriteLine("== weight byte stream (disk->ram bandwidth) ==");
using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
{
    fs.Seek((long)emb.Offset, SeekOrigin.Begin);
    var buf = new byte[1 << 20];
    ulong remaining = bytes;
    long total = 0;
    sw.Restart();
    while (remaining > 0)
    {
        int toRead = (int)Math.Min((ulong)buf.Length, remaining);
        int n = fs.Read(buf, 0, toRead);
        if (n <= 0) break;
        total += n;
        remaining -= (ulong)n;
    }
    sw.Stop();
    double gb = total / (1024.0 * 1024.0 * 1024.0);
    double sec = sw.Elapsed.TotalSeconds;
    Console.WriteLine($"read {gb:F2} GB in {sw.ElapsedMilliseconds} ms = {gb / sec:F2} GB/s");
}

Console.WriteLine("== int8 dot-product kernel (managed SIMD) ==");
BenchmarkInt8Dot();

Console.WriteLine("== dense decode estimate (Q4/Q6K, memory-bound) ==");
double paramsPerToken = 4.2e9;
double bytesPerToken = paramsPerToken * 0.78;
Console.WriteLine($"approx weight bytes touched per token: {bytesPerToken / 1e9:F2} GB");
Console.WriteLine("(full inference needs the native core compiled + gemma3 graph)");

return 0;

static void BenchmarkInt8Dot()
{
    const int n = 1 << 16;
    var a = new sbyte[n];
    var b = new sbyte[n];
    new Random(1).NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan()));
    new Random(1).NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(b.AsSpan()));

    long checksum = 0;
    var swatch = Stopwatch.StartNew();
    const int iters = 2000;
    for (int it = 0; it < iters; it++)
    {
        checksum += Dot8Simd(a, b, n);
    }
    swatch.Stop();
    double ops = (double)iters * n;
    double gops = ops / swatch.Elapsed.TotalSeconds / 1e9;
    Console.WriteLine($"int8 dot: {gops:F1} G ops/s (checksum {checksum & 0xFF})");
}

static long Dot8Simd(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b, int n)
{
    if (Avx2.IsSupported && n % 32 == 0)
    {
        unsafe
        {
            fixed (sbyte* pa = a)
            fixed (sbyte* pb = b)
            {
                var acc = Vector256<int>.Zero;
                for (int i = 0; i < n; i += 32)
                {
                    var va = Avx.LoadVector256(pa + i);
                    var vb = Avx.LoadVector256(pb + i);
                    var loA = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(va, 0));
                    var loB = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(vb, 0));
                    acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(loA, loB));
                    var hiA = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(va, 1));
                    var hiB = Avx2.ConvertToVector256Int16(Avx2.ExtractVector128(vb, 1));
                    acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(hiA, hiB));
                }
                var v0 = Sse2.Add(Avx2.ExtractVector128(acc, 0), Avx2.ExtractVector128(acc, 1));
                var tmp = stackalloc int[4];
                Sse2.Store(tmp, v0);
                return (long)tmp[0] + tmp[1] + tmp[2] + tmp[3];
            }
        }
    }
    return 0;
}