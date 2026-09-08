using DesireeIA.Formats;

if (args.Length < 1)
{
    Console.Error.WriteLine("usage: desireeia_inspect <model.gguf> [--tensors]");
    return 1;
}

var path = args[0];
var showTensors = args.Contains("--tensors");

var reader = new GgufReader();
var meta = reader.Open(path);

Console.WriteLine($"file: {path}");
Console.WriteLine($"version: {meta.Version}");
Console.WriteLine($"architecture: {meta.Architecture}");
Console.WriteLine($"context length: {meta.ContextLength}");
Console.WriteLine($"vocabulary size: {meta.VocabularySize}");
Console.WriteLine($"layers: {meta.LayerCount}");
Console.WriteLine($"embedding size: {meta.EmbeddingSize}");
Console.WriteLine($"embedding tensor type: {meta.EmbeddingType}");
Console.WriteLine($"metadata entries: {meta.KeyValueCount}");
Console.WriteLine($"tensor count: {meta.TensorCount}");

if (showTensors)
{
    var byType = meta.Tensors
        .GroupBy(t => t.Type)
        .OrderByDescending(g => g.Count())
        .Select(g => $"  {g.Key}: {g.Count()}");
    Console.WriteLine("tensors by type:");
    Console.WriteLine(string.Join(Environment.NewLine, byType));

    Console.WriteLine("first 20 tensors:");
    foreach (var t in meta.Tensors.Take(20))
    {
        Console.WriteLine($"  {t.Name} [{t.Type}] shape={string.Join('x', t.Shape)} offset={t.Offset}");
    }
}

return 0;