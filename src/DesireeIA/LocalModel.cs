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