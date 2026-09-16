// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DesireeIA.Native;

namespace DesireeIA;

/// <summary>Corresponds to desireeia_special_token in abi.h.</summary>
public enum SpecialToken
{
    Bos = 0,
    Eos = 1,
    Unk = 2,
    Pad = 3
}

/// <summary>
/// Sampling parameters. The defaults reproduce the previous behavior
/// (greedy, deterministic): raise <see cref="Temperature"/> to get sampled
/// generation.
/// </summary>
public sealed class SamplingOptions
{
    /// <summary>0 or less = greedy (argmax), deterministic.</summary>
    public float Temperature { get; init; }

    /// <summary>Consider only the K most probable tokens. 0 or less = disabled.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Cumulative probability mass to keep. 1 or more = disabled.</summary>
    public float TopP { get; init; } = 0.95f;

    /// <summary>
    /// Penalty on tokens that appeared recently. 1 = disabled.
    /// Helps avoid the repetitive loops that greedy decoding tends to fall
    /// into on long generations.
    /// </summary>
    public float PenaltyRepeat { get; init; } = 1.0f;

    public float PenaltyFrequency { get; init; }
    public float PenaltyPresence { get; init; }

    /// <summary>How many tokens back to look at for the penalties.</summary>
    public int PenaltyLastN { get; init; } = 64;

    /// <summary>Generator seed. 0 = random seed.</summary>
    public uint Seed { get; init; }
}

public sealed class LocalModel : IDisposable
{
    private readonly IntPtr _context;
    private bool _disposed;
    private readonly NativeMethods.LogCallback? _loggerInstance;

    public string ModelPath { get; }
    public ExecutionPlan Plan { get; }

    private LocalModel(IntPtr context, string modelPath, ExecutionPlan plan, NativeMethods.LogCallback? logger)
    {
        _context = context;
        ModelPath = modelPath;
        Plan = plan;
        _loggerInstance = logger;
    }

    public static LocalModel Load(string modelPath, ExecutionPlan plan, Action<string>? logger = null)
    {
        NativeMethods.LogCallback? cb = null;
        if (logger is not null)
        {
            cb = (level, msg, _) =>
            {
                var text = Marshal.PtrToStringUTF8(msg);
                if (text is not null)
                {
                    logger(text);
                }
            };
        }

        var err = NativeMethods.desireeia_create(modelPath, plan.ToNative(), cb, IntPtr.Zero, out var ctx);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Model load failed: {err}");
        }

        return new LocalModel(ctx, modelPath, plan, cb);
    }

    public int Predict(int[] tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_predict(_context, tokens, (nuint)tokens.Length, out var token);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Predict failed: {err}");
        }
        return token;
    }

    public int NextToken()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_next_token(_context, out var token);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Next token failed: {err}");
        }
        return token;
    }

    public ulong ContextSize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return NativeMethods.desireeia_context_size(_context);
    }

    /// <summary>
    /// The model's own trained/declared max context length (GGUF metadata
    /// key "&lt;arch&gt;.context_length"), or 0 if unknown for this model's
    /// architecture. Informational only: the engine's KV cache grows
    /// dynamically as tokens are processed and is NOT capped by this value -
    /// callers that want a fixed context window (e.g. to mirror another
    /// engine's behavior, or to budget a UI around it) should read this once
    /// after loading and enforce it themselves.
    /// </summary>
    public uint TrainedContextLength()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return NativeMethods.desireeia_context_length_trained(_context);
    }

    /// <summary>
    /// Tokenizes the text with the model's tokenizer (SentencePiece or BPE,
    /// auto-detected). Returns null if the model does not have a recognized
    /// tokenizer (see <see cref="HasTokenizer"/>).
    /// </summary>
    public int[]? Tokenize(string text, bool addBos = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_tokenize(_context, text, addBos ? 1 : 0, null, 0, out var count);
        if (err == NativeMethods.Error.NotSupported) return null;
        if (err != NativeMethods.Error.Ok) throw new InvalidOperationException($"Tokenize failed: {err}");

        var ids = new int[count];
        if (count > 0)
        {
            err = NativeMethods.desireeia_tokenize(_context, text, addBos ? 1 : 0, ids, count, out _);
            if (err != NativeMethods.Error.Ok) throw new InvalidOperationException($"Tokenize failed: {err}");
        }
        return ids;
    }

    /// <summary>
    /// Returns the token's text, or null if the model does not have a
    /// recognized tokenizer or the id is not valid.
    /// </summary>
    public string? TokenPiece(int id)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var buf = new byte[256];
        var err = NativeMethods.desireeia_token_piece(_context, id, buf, (nuint) buf.Length);
        if (err != NativeMethods.Error.Ok) return null;
        var len = Array.IndexOf(buf, (byte) 0);
        return System.Text.Encoding.UTF8.GetString(buf, 0, len < 0 ? buf.Length : len);
    }

    public bool HasTokenizer => Tokenize(string.Empty, addBos: false) is not null;

    /// <summary>
    /// BERT encoder: per-token embedding (not aggregated/normalized — pooling
    /// is left as an application choice). Returns null if the loaded model
    /// is not a BERT encoder. Each row of the result corresponds to a token
    /// in order, width = the model's embedding dimension.
    /// </summary>
    public float[][]? Embed(int[] tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.Length == 0) return Array.Empty<float[]>();
        var err = NativeMethods.desireeia_embed(_context, tokens, (nuint) tokens.Length,
            null, 0, out var len, out var dim);
        if (err == NativeMethods.Error.NotSupported) return null;
        if (err != NativeMethods.Error.Ok) throw new InvalidOperationException($"Embed failed: {err}");

        var flat = new float[len];
        if (len > 0)
        {
            err = NativeMethods.desireeia_embed(_context, tokens, (nuint) tokens.Length,
                flat, (nuint) flat.Length, out _, out dim);
            if (err != NativeMethods.Error.Ok) throw new InvalidOperationException($"Embed failed: {err}");
        }

        var rows = new float[tokens.Length][];
        for (int i = 0; i < tokens.Length; i++)
        {
            rows[i] = new float[dim];
            Array.Copy(flat, i * (int) dim, rows[i], 0, (int) dim);
        }
        return rows;
    }

    /// <summary>
    /// Id of the requested special token (BOS/EOS/UNK/PAD), or null if the
    /// model does not define it or does not have a recognized vocabulary.
    /// </summary>
    public int? SpecialTokenId(SpecialToken which)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_special_token_id(_context, (int) which, out var id);
        if (err != NativeMethods.Error.Ok) return null;
        return id < 0 ? null : id;
    }

    /// <summary>
    /// Id of the end-of-sequence token (EOS), or null if undefined. Used by
    /// callers that generate text to know when to stop (see Phase 0,
    /// docs/engine_gap_analysis.md: before this, the generate loop never
    /// stopped on its own, it always ran to --max-tokens).
    /// </summary>
    public int? EosId => SpecialTokenId(SpecialToken.Eos);

    /// <summary>
    /// True if the token marks END OF GENERATION. Comparing against
    /// <see cref="EosId"/> alone is not enough: chat models close the turn
    /// with dedicated tokens (gemma uses &lt;end_of_turn&gt;), and stopping
    /// only on EOS leaves the model repeating them indefinitely after it has
    /// answered.
    /// </summary>
    public bool IsEndOfGeneration(int tokenId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_is_eog_token(_context, tokenId, out var isEog);
        return err == NativeMethods.Error.Ok && isEog != 0;
    }

    /// <summary>
    /// Applies the chat format detected for the model (from the GGUF's
    /// tokenizer.chat_template, or a per-architecture default) to a sequence
    /// of messages (role, content), producing the prompt to tokenize. Covers
    /// the most widespread model families (ChatML/Qwen, Llama
    /// 2/3/3.1/3.2/3.3/4, Mistral in all its historical variants, Gemma 2/3,
    /// Phi 3/4, DeepSeek V2/V3/R1, Command-R, ChatGLM3/4, MiniCPM, Zephyr,
    /// Falcon3, Exaone3) — no longer just gemma hardcoded here in the CLI.
    /// Unrecognized formats fall back to ChatML.
    /// </summary>
    public string ApplyChatTemplate(IReadOnlyList<(string Role, string Content)> messages, bool addAssistant = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // string[] with ArraySubType=LPUTF8Str doesn't marshal on this
        // runtime: NUL-terminated UTF-8 pointers are allocated by hand (the
        // way Marshal.StringToCoTaskMemUTF8 would, rewritten here to control
        // the encoding explicitly) and passed as IntPtr[].
        var roleHandles = new IntPtr[messages.Count];
        var contentHandles = new IntPtr[messages.Count];
        try
        {
            for (int i = 0; i < messages.Count; i++)
            {
                roleHandles[i] = Utf8ZAlloc(messages[i].Role);
                contentHandles[i] = Utf8ZAlloc(messages[i].Content);
            }

            var err = NativeMethods.desireeia_apply_chat_template(
                _context, roleHandles, contentHandles, (nuint) messages.Count, addAssistant ? 1 : 0,
                null, 0, out var neededLen);
            if (err != NativeMethods.Error.Ok)
            {
                throw new InvalidOperationException($"Apply chat template failed: {err}");
            }

            var buf = new byte[neededLen + 1];
            err = NativeMethods.desireeia_apply_chat_template(
                _context, roleHandles, contentHandles, (nuint) messages.Count, addAssistant ? 1 : 0,
                buf, (nuint) buf.Length, out neededLen);
            if (err != NativeMethods.Error.Ok)
            {
                throw new InvalidOperationException($"Apply chat template failed: {err}");
            }
            return System.Text.Encoding.UTF8.GetString(buf, 0, (int) neededLen);
        }
        finally
        {
            foreach (var h in roleHandles) if (h != IntPtr.Zero) Marshal.FreeHGlobal(h);
            foreach (var h in contentHandles) if (h != IntPtr.Zero) Marshal.FreeHGlobal(h);
        }
    }

    private static IntPtr Utf8ZAlloc(string s)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(s);
        var p = Marshal.AllocHGlobal(bytes.Length + 1);
        Marshal.Copy(bytes, 0, p, bytes.Length);
        Marshal.WriteByte(p, bytes.Length, 0);
        return p;
    }

    /// <summary>
    /// Sets the sampling parameters. Takes effect from the next token onward.
    /// </summary>
    public void SetSampling(SamplingOptions options)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(options);
        var native = new NativeMethods.Sampling
        {
            Temperature    = options.Temperature,
            TopK           = options.TopK,
            TopP           = options.TopP,
            PenaltyRepeat  = options.PenaltyRepeat,
            PenaltyFreq    = options.PenaltyFrequency,
            PenaltyPresent = options.PenaltyPresence,
            PenaltyLastN   = options.PenaltyLastN,
            Seed           = options.Seed
        };
        var err = NativeMethods.desireeia_set_sampling(_context, in native);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Set sampling failed: {err}");
        }
    }

    /// <summary>Current sampling parameters.</summary>
    public SamplingOptions GetSampling()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_get_sampling(_context, out var n);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Get sampling failed: {err}");
        }
        return new SamplingOptions
        {
            Temperature      = n.Temperature,
            TopK             = n.TopK,
            TopP             = n.TopP,
            PenaltyRepeat    = n.PenaltyRepeat,
            PenaltyFrequency = n.PenaltyFreq,
            PenaltyPresence  = n.PenaltyPresent,
            PenaltyLastN     = n.PenaltyLastN,
            Seed             = n.Seed
        };
    }

    /// <summary>
    /// Loads a Recover-LoRA adapter in GGUF format (desireeialmn convention:
    /// tensors "&lt;base_name&gt;.lora_a"/".lora_b", metadata
    /// "adapter.lora.alpha"), applied at runtime without touching the
    /// quantized base weights. Can be called multiple times to load several
    /// adapters at once: their contributions add up. Covers attention,
    /// dense/MLA/shared-expert FFN; does not cover routed MoE experts. Must
    /// be called before the first generation to apply to every layer of a
    /// model with weight-cache enabled (layers already generated are still
    /// reloaded automatically on their next access, but only from that point
    /// onward).
    /// </summary>
    public void LoadLoraAdapter(string loraGgufPath, float scale = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_load_lora_adapter(_context, loraGgufPath, scale);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Load LoRA adapter failed: {err}");
        }
    }

    /// <summary>Removes all LoRA adapters loaded on the context.</summary>
    public void ClearLoraAdapters()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_clear_lora_adapters(_context);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Clear LoRA adapters failed: {err}");
        }
    }

    /// <summary>
    /// Loads a prerouter routing-prediction head (GGUF format, a convention
    /// specific to this engine: tensors
    /// "prerouter.&lt;N&gt;.fc1/fc2/linear_init.weight" for "owner" layer
    /// N). Predicts which experts layer N+1 will route to from layer N's
    /// data, so their weights can be prefetched in the background one layer
    /// ahead of the router's actual computation. Never affects correctness:
    /// a wrong or missing prediction simply leaves the existing read path
    /// running as-is. No trained head is publicly available today for any
    /// supported model: see <see cref="SetPrerouterHeuristic"/> for a
    /// heuristic fallback that requires no file at all.
    /// </summary>
    public void LoadPrerouter(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_load_prerouter(_context, path);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Load prerouter failed: {err}");
        }
    }

    /// <summary>Removes all loaded prerouter heads.</summary>
    public void ClearPrerouter()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_clear_prerouter(_context);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Clear prerouter failed: {err}");
        }
    }

    /// <summary>
    /// Enables/disables the fallback heuristic ("layer N+1 routes to the
    /// same experts layer N just used") for owner layers without a trained
    /// prerouter head loaded. Off by default; deliberately a placeholder,
    /// not an accurate prediction.
    /// </summary>
    public void SetPrerouterHeuristic(bool enabled)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_set_prerouter_heuristic(_context, enabled ? 1 : 0);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Set prerouter heuristic failed: {err}");
        }
    }

    // ============================================================
    // Streaming generation (async, with stop sequences)
    // ============================================================

    /// <summary>
    /// Generates in streaming fashion from an already-tokenized prompt,
    /// returning text pieces as soon as they're decoded and stopping at
    /// EOS/end-of-turn, at the <see cref="GenerateOptions.MaxTokens"/> limit,
    /// or on the first occurrence of one of the
    /// <see cref="GenerateOptions.StopSequences"/> (which is not included in
    /// the output, as in Ollama/OpenAI-style APIs). Calls into the native
    /// engine are synchronous (a per-ctx mutex serializes them anyway): the
    /// await Task.Yield() between one token and the next is there to leave
    /// the caller free to interleave other async work and observe
    /// cancellation token by token, not to run inference on another thread.
    /// </summary>
    public async IAsyncEnumerable<string> StreamAsync(int[] promptTokens, GenerateOptions? options = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new GenerateOptions();
        var scanner = new StopSequenceScanner(options.StopSequences);

        var token = Predict(promptTokens);
        for (int i = 0; i < options.MaxTokens; i++)
        {
            ct.ThrowIfCancellationRequested();
            if (IsEndOfGeneration(token)) break;

            var piece = TokenPiece(token) ?? "";
            var (emit, stopped) = scanner.Feed(piece);
            if (emit.Length > 0) yield return emit;
            if (stopped) yield break;

            await Task.Yield();
            token = NextToken();
        }

        var tail = scanner.Flush();
        if (tail.Length > 0) yield return tail;
    }

    /// <summary>
    /// Like <see cref="StreamAsync"/> but starting from a conversation
    /// (role, content): applies the model's chat template and tokenizes the
    /// resulting prompt before generating.
    /// </summary>
    public IAsyncEnumerable<string> ChatStreamAsync(IReadOnlyList<(string Role, string Content)> messages,
        GenerateOptions? options = null, bool addAssistant = true, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var prompt = ApplyChatTemplate(messages, addAssistant);
        var ids = Tokenize(prompt, addBos: true)
            ?? throw new InvalidOperationException("Model has no recognized tokenizer");
        return StreamAsync(ids, options, ct);
    }

    // ============================================================
    // Vision Module - Image/Video/File Stream Support
    // ============================================================

    /// <summary>
    /// Load an image from disk. Supports PNG, JPEG, BMP formats.
    /// The caller must free the returned image with FreeImage() when done.
    /// </summary>
    public static VisionImageWrapper LoadImage(string path, int expectedChannels = 3)
    {
        var err = NativeMethods.desireeia_load_image(path, expectedChannels, out var img);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Failed to load image: {err}");
        }
        return new VisionImageWrapper(img);
    }

    /// <summary>
    /// True if the loaded model has a vision encoder built into the GGUF
    /// (CLIP/SigLIP codec with clip.vision.*).
    /// </summary>
    public bool HasVision
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var err = NativeMethods.desireeia_has_vision(_context, out var has);
            if (err != NativeMethods.Error.Ok) return false;
            return has != 0;
        }
    }

    /// <summary>
    /// Encode an image into embedding vectors using the model's own vision
    /// encoder. Returns null if the model has no vision encoder.
    /// </summary>
    public float[]? EncodeImage(VisionImageWrapper image, out uint embeddingDim)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        embeddingDim = 0;

        var img = image.GetNative();
        var err = NativeMethods.desireeia_vision_encode_ctx(_context, in img,
            null, 0, out var len, out embeddingDim);
        if (err == NativeMethods.Error.NotSupported) return null;
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Vision encode failed: {err}");
        }

        var embd = new float[len];
        if (len > 0)
        {
            err = NativeMethods.desireeia_vision_encode_ctx(_context, in img,
                embd, (nuint)embd.Length, out _, out embeddingDim);
            if (err != NativeMethods.Error.Ok)
            {
                throw new InvalidOperationException($"Vision encode failed: {err}");
            }
        }
        return embd;
    }

    /// <summary>Number of vision embedding vectors per image.</summary>
    public int VisionTokenCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var err = NativeMethods.desireeia_vision_token_count(_context, out var n);
            if (err != NativeMethods.Error.Ok) return 0;
            return n;
        }
    }

    /// <summary>Id of the image placeholder token, or null if it cannot be resolved.</summary>
    public int? VisionImageToken
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var err = NativeMethods.desireeia_vision_image_token(_context, out var id);
            if (err != NativeMethods.Error.Ok || id < 0) return null;
            return id;
        }
    }

    /// <summary>
    /// Multimodal prefill: identical to <see cref="Predict"/> but the token
    /// stream must contain <see cref="VisionTokenCount"/> occurrences of the
    /// placeholder token, and <paramref name="embd"/> must be the result of
    /// <see cref="EncodeImage"/> (the vision embeddings, in order). Pass
    /// imageToken = -1 to use the auto-resolved placeholder.
    /// </summary>
    public int PredictWithImage(int[] tokens, float[] embd, int imageToken = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!HasVision) throw new InvalidOperationException("Model has no vision encoder");
        var err = NativeMethods.desireeia_predict_image(_context, tokens,
            (nuint)tokens.Length, embd, (nuint)embd.Length, imageToken, out var token);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"PredictWithImage failed: {err}");
        }
        return token;
    }

    /// <summary>
    /// Preprocess an image for vision encoder: resize to target_size,
    /// normalize to [0,1], convert to RGB. Returns flat CHW array.
    /// </summary>
    public static float[] PreprocessImage(VisionImageWrapper image, int targetSize)
    {
        var img = image.GetNative();
        var err = NativeMethods.desireeia_vision_preprocess(in img, targetSize,
            null, 0, out var len);
        if (err != NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Vision preprocess failed: {err}");
        }

        var pixels = new float[len];
        if (len > 0)
        {
            err = NativeMethods.desireeia_vision_preprocess(in img, targetSize,
                pixels, (nuint)pixels.Length, out _);
            if (err != NativeMethods.Error.Ok)
            {
                throw new InvalidOperationException($"Vision preprocess failed: {err}");
            }
        }
        return pixels;
    }

    /// <summary>
    /// Create a message with image data for multimodal models.
    /// The image is encoded as base64 and injected into the content.
    /// This works with models that accept image tokens in the chat template.
    /// </summary>
    public static string CreateImageMessage(string role, VisionImageWrapper image,
                                            string? textContent = null)
    {
        var img = image.GetNative();
        var bytes = new byte[img.Width * img.Height * img.Channels];
        if (img.Data != IntPtr.Zero && bytes.Length > 0)
        {
            System.Runtime.InteropServices.Marshal.Copy(img.Data, bytes, 0, bytes.Length);
        }
        var b64 = Convert.ToBase64String(bytes);
        var ext = img.Channels == 4 ? "rgba" : "rgb";
        var content = $"[image:{img.Width}x{img.Height}@{ext}:{b64}]";
        if (!string.IsNullOrEmpty(textContent))
        {
            content += "\n" + textContent;
        }
        return content;
    }

    /// <summary>
    /// Check if the last assistant response contains generated image data.
    /// Returns the image bytes if found, null otherwise.
    /// </summary>
    public static byte[]? ExtractGeneratedImage(string response)
    {
        // Look for base64-encoded image data in the response
        var b64Start = response.IndexOf("[generated_image:");
        if (b64Start < 0) return null;
        b64Start = response.IndexOf(':', b64Start) + 1;
        var b64End = response.IndexOf(']', b64Start);
        if (b64End < 0) return null;
        var b64 = response.Substring(b64Start, b64End - b64Start);
        return Convert.FromBase64String(b64);
    }

    /// <summary>
    /// Check if the last assistant response contains generated file data.
    /// Returns the file bytes and filename if found, null otherwise.
    /// </summary>
    public static (byte[] Data, string Filename)? ExtractGeneratedFile(string response)
    {
        var marker = "[generated_file:";
        var b64Start = response.IndexOf(marker);
        if (b64Start < 0) return null;
        b64Start += marker.Length;
        var colonPos = response.IndexOf(':', b64Start);
        if (colonPos < 0) return null;
        var filename = response.Substring(b64Start, colonPos - b64Start);
        b64Start = colonPos + 1;
        var b64End = response.IndexOf(']', b64Start);
        if (b64End < 0) return null;
        var b64 = response.Substring(b64Start, b64End - b64Start);
        return (Convert.FromBase64String(b64), filename);
    }

    /// <summary>
    /// Check if the last assistant response contains generated video data.
    /// Returns the video bytes and format if found, null otherwise.
    /// </summary>
    public static (byte[] Data, string Format)? ExtractGeneratedVideo(string response)
    {
        var marker = "[generated_video:";
        var b64Start = response.IndexOf(marker);
        if (b64Start < 0) return null;
        b64Start += marker.Length;
        var colonPos = response.IndexOf(':', b64Start);
        if (colonPos < 0) return null;
        var format = response.Substring(b64Start, colonPos - b64Start);
        b64Start = colonPos + 1;
        var b64End = response.IndexOf(']', b64Start);
        if (b64End < 0) return null;
        var b64 = response.Substring(b64Start, b64End - b64Start);
        return (Convert.FromBase64String(b64), format);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_context != IntPtr.Zero)
        {
            NativeMethods.desireeia_destroy(_context);
        }
        GC.SuppressFinalize(this);
    }

    ~LocalModel() => Dispose();
}