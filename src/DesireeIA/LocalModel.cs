using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DesireeIA.Native;

namespace DesireeIA;

/// <summary>Corrisponde a desireeia_special_token in abi.h.</summary>
public enum SpecialToken
{
    Bos = 0,
    Eos = 1,
    Unk = 2,
    Pad = 3
}

/// <summary>
/// Parametri di campionamento. I default riproducono il comportamento
/// precedente (greedy, deterministico): serve alzare <see cref="Temperature"/>
/// per avere generazione campionata.
/// </summary>
public sealed class SamplingOptions
{
    /// <summary>0 o meno = greedy (argmax), deterministico.</summary>
    public float Temperature { get; init; }

    /// <summary>Considera solo i K token piu' probabili. 0 o meno = disattivato.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Massa di probabilita' cumulata da tenere. 1 o piu' = disattivato.</summary>
    public float TopP { get; init; } = 0.95f;

    /// <summary>
    /// Penalita' sui token gia' comparsi di recente. 1 = disattivata.
    /// Serve a evitare i cicli ripetitivi in cui la decodifica greedy
    /// finisce su generazioni lunghe.
    /// </summary>
    public float PenaltyRepeat { get; init; } = 1.0f;

    public float PenaltyFrequency { get; init; }
    public float PenaltyPresence { get; init; }

    /// <summary>Quanti token indietro guardare per le penalita'.</summary>
    public int PenaltyLastN { get; init; } = 64;

    /// <summary>Seme del generatore. 0 = seme casuale.</summary>
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
    /// Tokenizza il testo con il tokenizer del modello (SentencePiece o BPE,
    /// auto-rilevato). Restituisce null se il modello non ha un tokenizer
    /// riconosciuto (vedi <see cref="HasTokenizer"/>).
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
    /// Restituisce il testo del token, o null se il modello non ha un
    /// tokenizer riconosciuto o l'id non e' valido.
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
    /// Encoder BERT: embedding per token (non aggregato/normalizzato — il
    /// pooling e' una scelta dell'applicazione). Restituisce null se il
    /// modello caricato non e' un encoder BERT. Ogni riga del risultato
    /// corrisponde a un token in ordine, larghezza = dimensione embedding
    /// del modello.
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
    /// Id del token speciale richiesto (BOS/EOS/UNK/PAD), o null se il
    /// modello non lo definisce o non ha un vocabolario riconosciuto.
    /// </summary>
    public int? SpecialTokenId(SpecialToken which)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_special_token_id(_context, (int) which, out var id);
        if (err != NativeMethods.Error.Ok) return null;
        return id < 0 ? null : id;
    }

    /// <summary>
    /// Id del token di fine sequenza (EOS), o null se non definito. Usato
    /// da chi genera per sapere quando fermarsi (vedi Fase 0,
    /// docs/engine_gap_analysis.md: prima di questo il ciclo di generate
    /// non si fermava mai da solo, andava sempre a --max-tokens).
    /// </summary>
    public int? EosId => SpecialTokenId(SpecialToken.Eos);

    /// <summary>
    /// True se il token e' di FINE GENERAZIONE. Non basta confrontare con
    /// <see cref="EosId"/>: i modelli chat chiudono il turno con token
    /// dedicati (gemma usa &lt;end_of_turn&gt;), e fermarsi solo su EOS
    /// lascia il modello a ripeterli all'infinito dopo aver risposto.
    /// </summary>
    public bool IsEndOfGeneration(int tokenId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var err = NativeMethods.desireeia_is_eog_token(_context, tokenId, out var isEog);
        return err == NativeMethods.Error.Ok && isEog != 0;
    }

    /// <summary>
    /// Applica il formato di chat rilevato per il modello (dal
    /// tokenizer.chat_template del GGUF, o da un default per architettura)
    /// a una sequenza di messaggi (role, content), producendo il prompt da
    /// tokenizzare. Copre le famiglie di modelli piu' diffuse (ChatML/Qwen,
    /// Llama 2/3/3.1/3.2/3.3/4, Mistral in tutte le varianti storiche,
    /// Gemma 2/3, Phi 3/4, DeepSeek V2/V3/R1, Command-R, ChatGLM3/4,
    /// MiniCPM, Zephyr, Falcon3, Exaone3) — non piu' solo gemma cablata qui
    /// nella CLI. Formati non riconosciuti ricadono su ChatML.
    /// </summary>
    public string ApplyChatTemplate(IReadOnlyList<(string Role, string Content)> messages, bool addAssistant = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // string[] con ArraySubType=LPUTF8Str non marshala su questo
        // runtime: si allocano a mano puntatori UTF-8 NUL-terminati (come
        // farebbe Marshal.StringToCoTaskMemUTF8, riscritto qui per
        // controllare l'encoding esplicitamente) e si passano come IntPtr[].
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
    /// Imposta i parametri di campionamento. Ha effetto dal token successivo.
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

    /// <summary>Parametri di campionamento correnti.</summary>
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
    /// Carica un adapter Recover-LoRA in formato GGUF (convenzione
    /// desireeialmn: tensori "&lt;nome_base&gt;.lora_a"/".lora_b", metadato
    /// "adapter.lora.alpha"), applicato a runtime senza toccare i pesi base
    /// quantizzati. Puo' essere chiamato piu' volte per caricare piu'
    /// adapter contemporaneamente: i contributi si sommano. Copre
    /// attenzione, FFN densa/MLA/shared-expert; non copre gli esperti MoE
    /// instradati. Va chiamato prima della prima generazione per applicarsi
    /// a tutti i layer di un modello con weight-cache abilitata (layer gia'
    /// generati vengono comunque ricaricati automaticamente al prossimo
    /// accesso, ma solo da quel punto in poi).
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

    /// <summary>Rimuove tutti gli adapter LoRA caricati sul contesto.</summary>
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
    /// Carica una testa di previsione del routing prerouter (formato GGUF,
    /// convenzione propria di questo motore: tensori
    /// "prerouter.&lt;N&gt;.fc1/fc2/linear_init.weight" per layer "owner"
    /// N). Predice quali esperti instradera' il layer N+1 dai dati del
    /// layer N, cosi' i loro pesi possono essere precaricati in background
    /// un layer prima del calcolo reale del router. Non ha mai effetto
    /// sulla correttezza: una previsione sbagliata o assente lascia
    /// semplicemente girare il percorso di lettura gia' esistente.
    /// Nessuna testa addestrata e' oggi disponibile pubblicamente per
    /// nessun modello supportato: vedi <see cref="SetPrerouterHeuristic"/>
    /// per un fallback euristico che non richiede alcun file.
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

    /// <summary>Rimuove tutte le teste prerouter caricate.</summary>
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
    /// Attiva/disattiva l'euristica di fallback ("il layer N+1 instrada
    /// agli stessi esperti appena usati dal layer N") per i layer owner
    /// senza una testa prerouter addestrata caricata. Spenta di default;
    /// dichiaratamente un placeholder, non una previsione accurata.
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
    // Streaming generation (async, con stop sequences)
    // ============================================================

    /// <summary>
    /// Genera in streaming a partire da un prompt gia' tokenizzato,
    /// restituendo i pezzi di testo non appena decodificati e fermandosi a
    /// EOS/end-of-turn, al limite <see cref="GenerateOptions.MaxTokens"/>, o
    /// alla prima occorrenza di una delle <see cref="GenerateOptions.StopSequences"/>
    /// (che non viene incluse nell'output, come nelle API stile Ollama/OpenAI).
    /// Le chiamate al motore nativo sono sincrone (un mutex per ctx le
    /// serializza comunque): lo await Task.Yield() fra un token e il
    /// successivo serve a lasciare il chiamante libero di intercalare altro
    /// lavoro asincrono e osservare la cancellazione token per token, non a
    /// far girare l'inferenza su un altro thread.
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
    /// Come <see cref="StreamAsync"/> ma a partire da una conversazione
    /// (role, content): applica il chat template del modello e tokenizza il
    /// prompt risultante prima di generare.
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
    /// True se il modello caricato ha un encoder visivo (vision) integrato
    /// nel GGUF (codec CLIP/SigLIP con clip.vision.*).
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

    /// <summary>Numero di vector di embedding visivo per immagine.</summary>
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

    /// <summary>Id del token placeholder immagine, o null se non risolvibile.</summary>
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
    /// Prefill multimodale: identico a <see cref="Predict"/> ma il flusso
    /// token deve contenere <see cref="VisionTokenCount"/> occorrenze del
    /// token placeholder, e <paramref name="embd"/> deve essere il risultato
    /// di <see cref="EncodeImage"/> (le embeddings visive, in ordine).
    /// Passare imageToken = -1 per usare il placeholder auto-risolto.
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