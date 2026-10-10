// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using DesireeIA.Native;

namespace DesireeIA;

/// <summary>
/// Thin wrapper over the engine's context memory (desireeia_memory_* in
/// abi.h): an on-board RAG working store on SSD for big tool payloads, so
/// they cost neither prompt tokens nor KV cache in VRAM. Storage, Brotli
/// compression, indexing, hybrid search, write-through edits and session
/// expiry all run inside the native engine; this class only marshals
/// strings. Every call returns the engine's JSON as a <see cref="JsonDocument"/>.
/// </summary>
public sealed class ContextMemory : IDisposable
{
    private IntPtr _mem;

    public ContextMemory(string rootDir, double ttlSeconds = 1800)
    {
        Check(NativeMethods.desireeia_memory_open(Z(rootDir), ttlSeconds, out _mem), "open");
    }

    /// <summary>Use a loaded encoder model for the vector half of search; null = BM25 only.</summary>
    public void SetEmbedder(LocalModel? model) =>
        Check(NativeMethods.desireeia_memory_set_embedder(Live(), model?.NativeHandle ?? IntPtr.Zero), "set_embedder");

    public JsonDocument PutFile(string session, string name, string path) =>
        Json(NativeMethods.desireeia_memory_put_file(Live(), Z(session), Z(name), Z(path), out var j), j, "put_file");

    public JsonDocument PutText(string session, string name, string text)
    {
        var raw = Encoding.UTF8.GetBytes(text);
        return Json(NativeMethods.desireeia_memory_put_text(Live(), Z(session), Z(name), raw, (nuint)raw.Length, out var j), j, "put_text");
    }

    public JsonDocument Append(string session, string name, string text)
    {
        var raw = Encoding.UTF8.GetBytes(text);
        return Json(NativeMethods.desireeia_memory_append(Live(), Z(session), Z(name), raw, (nuint)raw.Length, out var j), j, "append");
    }

    public JsonDocument Read(string session, string handle, uint offset = 1, uint limit = 200) =>
        Json(NativeMethods.desireeia_memory_read(Live(), Z(session), Z(handle), offset, limit, out var j), j, "read");

    public JsonDocument Search(string session, string query, string? handle = null, uint k = 5) =>
        Json(NativeMethods.desireeia_memory_search(Live(), Z(session), Z(query), handle is null ? null : Z(handle), k, out var j), j, "search");

    public JsonDocument List(string session) =>
        Json(NativeMethods.desireeia_memory_list(Live(), Z(session), out var j), j, "list");

    public JsonDocument ReplaceLines(string session, string handle, uint first, uint last, string text)
    {
        var raw = Encoding.UTF8.GetBytes(text);
        return Json(NativeMethods.desireeia_memory_replace_lines(Live(), Z(session), Z(handle), first, last, raw, (nuint)raw.Length, out var j),
            j, "replace_lines");
    }

    public JsonDocument Export(string session, string handle, string path) =>
        Json(NativeMethods.desireeia_memory_export(Live(), Z(session), Z(handle), Z(path), out var j), j, "export");

    public void Touch(string session) => Check(NativeMethods.desireeia_memory_touch(Live(), Z(session)), "touch");

    public void DropSession(string session) => Check(NativeMethods.desireeia_memory_drop_session(Live(), Z(session)), "drop_session");

    public int Sweep()
    {
        Check(NativeMethods.desireeia_memory_sweep(Live(), out var removed), "sweep");
        return removed;
    }

    public void Dispose()
    {
        if (_mem != IntPtr.Zero)
        {
            NativeMethods.desireeia_memory_close(_mem);
            _mem = IntPtr.Zero;
        }
    }

    private IntPtr Live() => _mem != IntPtr.Zero ? _mem : throw new ObjectDisposedException(nameof(ContextMemory));

    private static byte[] Z(string s)
    {
        var bytes = new byte[Encoding.UTF8.GetByteCount(s) + 1];
        Encoding.UTF8.GetBytes(s, 0, s.Length, bytes, 0);
        return bytes;
    }

    private static void Check(NativeMethods.Error err, string what)
    {
        if (err != NativeMethods.Error.Ok)
            throw new InvalidOperationException($"context memory {what} failed: {err}");
    }

    private static JsonDocument Json(NativeMethods.Error err, IntPtr json, string what)
    {
        Check(err, what);
        try
        {
            return JsonDocument.Parse(Marshal.PtrToStringUTF8(json) ?? "{}");
        }
        finally
        {
            NativeMethods.desireeia_memory_free_string(json);
        }
    }
}
