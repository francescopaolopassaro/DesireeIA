using System.Diagnostics;
using System.Linq;
using DesireeIA;

if (args.Length == 0)
{
    PrintUsage();
    return 1;
}

var command = args[0];
var rest = args.Skip(1).ToArray();

try
{
    return command switch
    {
        "info" => CmdInfo(rest),
        "tokenize" => CmdTokenize(rest),
        "embed" => CmdEmbed(rest),
        "generate" => CmdGenerate(rest),
        "bench" => CmdBench(rest),
        "hw" => CmdHw(),
        _ => Unknown(command)
    };
}
catch (DllNotFoundException ex)
{
    Console.Error.WriteLine($"libreria nativa non trovata: {ex.Message}");
    Console.Error.WriteLine("compilare prima il motore nativo (cmake + compilatore C++17).");
    return 2;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"errore: {ex.Message}");
    return 1;
}

static int Unknown(string cmd)
{
    Console.Error.WriteLine($"comando sconosciuto: {cmd}");
    PrintUsage();
    return 1;
}

static void PrintUsage()
{
    Console.WriteLine("""
        DesireeIA CLI - client di test per il motore nativo

        Uso:
          desireeia-cli hw
          desireeia-cli info <modello.gguf>
          desireeia-cli tokenize <modello.gguf> "<testo>"
          desireeia-cli embed <modello.gguf> "<testo>"  (solo encoder BERT)
          desireeia-cli generate <modello.gguf> "<testo>" [--max-tokens N] [--chat]
                     [--temp T] [--top-k K] [--top-p P]
                     [--repeat-penalty R] [--repeat-last-n N] [--seed S]
          desireeia-cli bench <modello.gguf> [--tokens N] [--warmup N] [--prompt "<testo>"] [--threads N]

        Campionamento: senza --temp la generazione e' greedy (deterministica).
        --temp 0.8 --repeat-penalty 1.1 e' un punto di partenza ragionevole;
        la penalita' di ripetizione serve a evitare i cicli in cui la
        decodifica greedy finisce su generazioni lunghe.
        """);
}

static int CmdHw()
{
    var hw = DesireeIAEngine.DetectHardware();
    Console.WriteLine($"cpu_threads = {hw.CpuThreads}");
    Console.WriteLine($"avx/avx2/avx512/neon = {hw.Avx}/{hw.Avx2}/{hw.Avx512}/{hw.Neon}");
    Console.WriteLine($"ram_total_mb = {hw.RamTotalMb}   ram_free_mb = {hw.RamFreeMb}");
    Console.WriteLine($"cuda={hw.CudaDeviceCount} intel_arc={hw.IntelGpuCount} axelera={hw.AxeleraDeviceCount} " +
                       $"metal={hw.Metal} vulkan={hw.Vulkan}");
    return 0;
}

static (LocalModel model, ExecutionPlan plan) Open(string modelPath, int? threads = null)
{
    if (!File.Exists(modelPath))
    {
        throw new FileNotFoundException($"modello non trovato: {modelPath}");
    }
    var overrides = threads is int t ? new ExecutionPlan { ThreadCount = t } : null;
    var plan = DesireeIAEngine.BuildPlan(modelPath, overrides);
    var model = LocalModel.Load(modelPath, plan);
    return (model, plan);
}

static int CmdInfo(string[] rest)
{
    if (rest.Length < 1)
    {
        Console.Error.WriteLine("uso: desireeia-cli info <modello.gguf>");
        return 1;
    }

    var (model, plan) = Open(rest[0]);
    using (model)
    {
        Console.WriteLine($"path        = {model.ModelPath}");
        Console.WriteLine($"plan        = {plan}");
        Console.WriteLine($"tokenizer   = {(model.HasTokenizer ? "riconosciuto (SPM o BPE)" : "non riconosciuto")}");
        Console.WriteLine($"context_size = {model.ContextSize()} byte (0 prima del primo Predict)");
    }
    return 0;
}

static int CmdTokenize(string[] rest)
{
    if (rest.Length < 2)
    {
        Console.Error.WriteLine("uso: desireeia-cli tokenize <modello.gguf> \"<testo>\"");
        return 1;
    }

    var (model, _) = Open(rest[0]);
    using (model)
    {
        var ids = model.Tokenize(rest[1]);
        if (ids is null)
        {
            Console.Error.WriteLine("il modello non ha un tokenizer riconosciuto (ne' SentencePiece ne' BPE).");
            return 1;
        }
        foreach (var id in ids)
        {
            var piece = model.TokenPiece(id) ?? "?";
            Console.WriteLine($"{id}\t{piece}");
        }
        Console.WriteLine($"({ids.Length} token)");
    }
    return 0;
}

static int CmdEmbed(string[] rest)
{
    if (rest.Length < 2)
    {
        Console.Error.WriteLine("uso: desireeia-cli embed <modello.gguf> \"<testo>\"");
        return 1;
    }

    var (model, _) = Open(rest[0]);
    using (model)
    {
        var ids = model.Tokenize(rest[1]);
        if (ids is null)
        {
            Console.Error.WriteLine("il modello non ha un tokenizer riconosciuto.");
            return 1;
        }
        var rows = model.Embed(ids);
        if (rows is null)
        {
            Console.Error.WriteLine("il modello caricato non e' un encoder BERT (desireeia_embed non supportato).");
            return 1;
        }
        Console.WriteLine($"{ids.Length} token, dimensione embedding = {(rows.Length > 0 ? rows[0].Length : 0)}");

        // Mean-pooling: convenzione comune per un embedding di frase, MA e'
        // una scelta dell'applicazione (CLI), non del motore — desireeia_embed
        // restituisce sempre l'embedding grezzo per token.
        if (rows.Length == 0) return 0;
        var dim = rows[0].Length;
        var pooled = new float[dim];
        foreach (var row in rows)
            for (int i = 0; i < dim; i++) pooled[i] += row[i] / rows.Length;
        var norm = MathF.Sqrt(pooled.Sum(v => v * v));
        Console.WriteLine($"mean-pool (prime 8 componenti, norma L2={norm:F4}): " +
            string.Join(", ", pooled.Take(8).Select(v => v.ToString("F4"))));
    }
    return 0;
}

static int CmdGenerate(string[] rest)
{
    if (rest.Length < 2)
    {
        Console.Error.WriteLine("uso: desireeia-cli generate <modello.gguf> \"<testo>\" [--max-tokens N] [--chat]");
        return 1;
    }

    var maxTokens = GetIntOption(rest, "--max-tokens", 32);
    var useChat = HasFlag(rest, "--chat");
    var (model, _) = Open(rest[0]);
    using (model)
    {
        ApplySamplingOptions(model, rest);
        // Il formato di chat e' rilevato dal motore al caricamento (dal
        // tokenizer.chat_template del GGUF, o da un default per
        // architettura): prima era gemma cablata qui, unico formato
        // supportato — ora copre le famiglie di modelli piu' diffuse
        // (vedi LocalModel.ApplyChatTemplate).
        var promptText = useChat
            ? model.ApplyChatTemplate(new[] { ("user", rest[1]) })
            : rest[1];
        var ids = model.Tokenize(promptText);
        if (ids is null || ids.Length == 0)
        {
            Console.Error.WriteLine("tokenizzazione fallita o prompt vuoto (tokenizer non riconosciuto?).");
            return 1;
        }

        Console.Write(rest[1]);
        var next = model.Predict(ids);
        bool stopped = model.IsEndOfGeneration(next);
        if (!stopped)
        {
            Console.Write(model.TokenPiece(next) ?? "");
            for (int i = 1; i < maxTokens; i++)
            {
                next = model.NextToken();
                if (model.IsEndOfGeneration(next)) { stopped = true; break; }
                Console.Write(model.TokenPiece(next) ?? "");
            }
        }
        Console.WriteLine();
        Console.WriteLine();
        if (!stopped)
        {
            Console.WriteLine($"[generazione fermata da --max-tokens ({maxTokens}), non da un token di fine]");
        }
        var s = model.GetSampling();
        Console.WriteLine(s.Temperature > 0
            ? $"[campionamento: temp={s.Temperature} top-k={s.TopK} top-p={s.TopP} " +
              $"repeat-penalty={s.PenaltyRepeat} (ultimi {s.PenaltyLastN})]"
            : "[greedy (argmax): deterministico. Su generazioni lunghe entra in cicli " +
              "ripetitivi: usare --temp 0.8 --repeat-penalty 1.1]");
        Console.WriteLine("[NB: nessuna validazione di correttezza logit-per-logit contro un " +
                           "riferimento - vedi docs/engine_gap_analysis.md]");
    }
    return 0;
}

static int CmdBench(string[] rest)
{
    if (rest.Length < 1)
    {
        Console.Error.WriteLine("uso: desireeia-cli bench <modello.gguf> [--tokens N] [--warmup N] [--prompt \"<testo>\"] [--threads N]");
        return 1;
    }

    var nTokens = GetIntOption(rest, "--tokens", 32);
    var nWarmup = GetIntOption(rest, "--warmup", 2);
    var prompt = GetStringOption(rest, "--prompt", null);
    var threadsOpt = GetIntOption(rest, "--threads", 0);

    var (model, plan) = Open(rest[0], threadsOpt > 0 ? threadsOpt : null);
    using (model)
    {
        Console.WriteLine($"plan: {plan}");

        int[] promptIds;
        if (prompt is not null)
        {
            promptIds = model.Tokenize(prompt) ?? new[] { 0 };
        }
        else
        {
            var bosIds = model.Tokenize(string.Empty);
            promptIds = bosIds is { Length: > 0 } ? bosIds : new[] { 0 };
        }

        var sw = Stopwatch.StartNew();
        model.Predict(promptIds);
        sw.Stop();
        var prefillMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"prefill: {promptIds.Length} token in {prefillMs:F1} ms " +
                           $"({promptIds.Length / Math.Max(prefillMs / 1000.0, 1e-6):F2} tok/s)");

        for (int i = 0; i < nWarmup; i++) model.NextToken();

        DesireeIAEngine.ProfileReset();
        sw.Restart();
        for (int i = 0; i < nTokens; i++) model.NextToken();
        sw.Stop();
        var decodeMs = sw.Elapsed.TotalMilliseconds;
        var tokPerSec = nTokens / Math.Max(decodeMs / 1000.0, 1e-6);
        Console.WriteLine($"decode: {nTokens} token in {decodeMs:F1} ms " +
                           $"({tokPerSec:F2} tok/s, {decodeMs / Math.Max(nTokens, 1):F2} ms/token)");
        Console.WriteLine($"profile (solo decode, {nTokens} token): {DesireeIAEngine.ProfileDump()}");
        Console.WriteLine($"context_size dopo il bench: {model.ContextSize()} byte");
    }
    return 0;
}

// Campionamento: senza questi flag il comportamento resta greedy
// (deterministico), come prima che il sampling esistesse. --temp lo attiva.
static void ApplySamplingOptions(LocalModel model, string[] args)
{
    var temp = GetFloatOption(args, "--temp", 0.0f);
    var options = new SamplingOptions
    {
        Temperature      = temp,
        TopK             = GetIntOption(args, "--top-k", 40),
        TopP             = GetFloatOption(args, "--top-p", 0.95f),
        PenaltyRepeat    = GetFloatOption(args, "--repeat-penalty", 1.0f),
        PenaltyLastN     = GetIntOption(args, "--repeat-last-n", 64),
        Seed             = (uint) GetIntOption(args, "--seed", 0)
    };
    model.SetSampling(options);
}

static float GetFloatOption(string[] args, string name, float def)
{
    for (int i = 0; i < args.Length - 1; i++)
    {
        if (args[i] == name &&
            float.TryParse(args[i + 1], System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out var v))
        {
            return v;
        }
    }
    return def;
}

static int GetIntOption(string[] args, string name, int def)
{
    for (int i = 0; i < args.Length - 1; i++)
    {
        if (args[i] == name && int.TryParse(args[i + 1], out var v)) return v;
    }
    return def;
}

static string? GetStringOption(string[] args, string name, string? def)
{
    for (int i = 0; i < args.Length - 1; i++)
    {
        if (args[i] == name) return args[i + 1];
    }
    return def;
}

static bool HasFlag(string[] args, string name) => Array.IndexOf(args, name) >= 0;
