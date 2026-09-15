// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

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

if (args.Contains("--kv"))
{
    foreach (var kvp in reader.RawKeyValues.OrderBy(k => k.Key))
    {
        Console.WriteLine($"  {kvp.Key} = {kvp.Value}");
    }
}

if (showTensors)
{
    var byType = meta.Tensors
        .GroupBy(t => t.Type)
        .OrderByDescending(g => g.Count())
        .Select(g => $"  {g.Key}: {g.Count()}");
    Console.WriteLine("tensors by type:");
    Console.WriteLine(string.Join(Environment.NewLine, byType));

    // Optional substring filter after --tensors, so a specific layer or
    // tensor family can be inspected without dumping hundreds of lines.
    var idx = Array.IndexOf(args, "--tensors");
    var filter = (idx >= 0 && idx + 1 < args.Length && !args[idx + 1].StartsWith("--"))
        ? args[idx + 1] : null;
    var shown = meta.Tensors.Where(t => filter is null || t.Name.Contains(filter));
    Console.WriteLine(filter is null ? "first 20 tensors:" : $"tensors matching '{filter}':");
    foreach (var t in (filter is null ? shown.Take(20) : shown))
    {
        Console.WriteLine($"  {t.Name} [{t.Type}] shape={string.Join('x', t.Shape)} offset={t.Offset}");
    }
}

return 0;