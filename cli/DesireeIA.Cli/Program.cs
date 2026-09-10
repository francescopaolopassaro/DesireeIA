using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using DesireeIA;

const string LogoAscii =
    "                                   .....;X&$$XXX$&x...                                                \n" +
    "                                   .:;&X;.... ....:&X:.                                                \n" +
    "                                 ..+&+.....:+x;....:&;x&X:.                                            \n" +
    "                              ....&x....:$$;..:X$..XX....$$... ...                                      \n" +
    "                          ... ..;&:....+&:X&&&&&&&&&:.....:&:.  ..                                      \n" +
    "                   ..  . .... ..&;....;&&+. ...  ...+&:.....x.....                                     \n" +
    "                   .   ...    .&+.:...$x...         ..;..  .:&....                                     \n" +
    "                             .:&.:&:.;&.     .        :$:  ..$x;..                                     \n" +
    "                             .+X.+&..&+....    ........:x:. .+$.                                       \n" +
    "                .           ..xx:&..;&&&&$X..  .+$&&$$x.&+.  :x.                                       \n" +
    "                         .  .:xx$+..&;.......  ....:....$&;  .&.                                       \n" +
    "                             .X&x..$&&x&&xX.&...;X&&X$&:Xx&+..&:..         :..                         \n" +
    "                         .....&x.:&x........$:.  .......X&:X&:&:. ...                                  \n" +
    "                            .X$.X&&X..    ..$..        .&&X.;&&;..                                     \n" +
    "                          ..+$:&x&;&.     .......     .;&.&x.:&x..        ..                            \n" +
    "                          ..&$&.+$.&+     ..+$X;..    .X&$$&x.+&..       ...                            \n" +
    "                          ..$&..;$.x$.  ............  .&;.XX&:.&+::..                                  \n" +
    "                            $+...&..$X....+XxxxXX;....&x..Xx+$.:&:...                                  \n" +
    "                      .     $x. .+&..$&....;&&&$....:&+. .&:.&:.&x...                                  \n" +
    "                     ...    .&;...$X..;&X..........$&+...XX..&:.+$...                                  \n" +
    "                     . .    .:&+...$X..+&$$;:..:+&X.$:..+&..:&..;$...                                  \n" +
    "                             ..&X...XX.:&..:+xx+....&:.x$...&+..xX...                                  \n" +
    "             .                .X&$..:&;.&.         .Xx$x...XX...&;                                      \n" +
    "                              .Xxx&..XX:&.       ...+&:...$x...XX.     ..                               \n" +
    "                              .&::&:.;$x$..      ..x$.. +&:...X$..     ..                               \n" +
    "                            .;&;..&; x&&...      .+&. .$X....&x$&&;.  ...                              \n" +
    "                       ...:X&x...+&..&&:        .:&...Xx...+&......X&$:..                              \n" +
    "                       .+x:..   .x..:x..         :+...X.. .x..      ..x:.                              \n" +
    "                       ......    . .. ..          ... ... ...       .....                               \n" +
    "                                   ...;.                                                             \n" +
    "    .&&&&&&&&X...                  ..+$.                            .&&&.   ..&&&$.                      \n" +
    "    .&+.   ..+&+.....+x:.. ...:x+:...:;...;:.;+....+x:...  ..:x+.....&&&.  ..&&&&&x..                 \n" +
    "    .&+       +&:.+&x...&&..X&....x. x&...&&$...x&x...&$...&$...x&+..&&&....X&&.+&&+.                 \n" +
    "    .&+     ..:&;.&+:::::$X.X&x::. ..x&. .&x. ..&+:::::$X.$$:::::x&..&&&...x&&;..X&&:...             \n" +
    "    .&+     ..$&..&;...........:x$&:.x&. .&x   .&;........$X.........&&&..+&&&&&&&&&&:.               \n" +
    "    .&+...:;X&X...+&x:..+X..x+...+&;.x&. .&x  ..+&x...+X..:&$;..:X;..&&&.:&&X.....:&&&..             \n" +
    "    .++++++;:.     .:+++;.  .;+++;...:+...+:.    .:+++;.   ..;++;.  .;++.+++..     :++;.               \n" +
    "    ...   .. .     ......   . .  . ... ... ..    .......   .  ....   ..... ...    ......                \n";

static void ShowSplash()
{
    Console.Write(LogoAscii);
}

ShowSplash();

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
        "generate" => await CmdGenerate(rest),
        "bench" => CmdBench(rest),
        "chat" => CmdChat(rest),
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
                     [--stop "<s>"]...  (ripetibile: ferma la generazione alla prima occorrenza,
                                         esclusa dall'output - vedi GenerateOptions.StopSequences)
                     [--json [--schema "<json-schema>"]]  (best-effort: istruisce il modello a
                                         rispondere solo con JSON e ne estrae il blocco valido;
                                         non e' grammar-constrained decoding)
                     [--lora <adapter.gguf>] [--lora-scale S]  (Recover-LoRA, vedi README)
                     [--prerouter <file.gguf>] [--prerouter-heuristic]  (vedi README)
          desireeia-cli bench <modello.gguf> [--tokens N] [--warmup N] [--prompt "<testo>"] [--threads N]
                     [--lora <adapter.gguf>] [--lora-scale S]
                     [--prerouter <file.gguf>] [--prerouter-heuristic]
          desireeia-cli chat <modello.gguf> [--temp T] [--top-k K] [--top-p P] [--max-tokens N]
                     [--lora <adapter.gguf>] [--lora-scale S]
                     [--prerouter <file.gguf>] [--prerouter-heuristic]

        Chat session:
          desireeia-cli chat <modello.gguf>  (interactive multi-turn conversation)
          /image <path>   load image file and inject into context (base64)
          /save <path>    save last assistant response to file
          /saveb64 <path> decode last base64 block and save as binary file

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

static (LocalModel model, ExecutionPlan plan) Open(string modelPath, int? threads = null, string[]? args = null)
{
    if (!File.Exists(modelPath))
    {
        throw new FileNotFoundException($"modello non trovato: {modelPath}");
    }
    var overrides = threads is int t ? new ExecutionPlan { ThreadCount = t } : null;
    var plan = DesireeIAEngine.BuildPlan(modelPath, overrides);
    Action<string>? logger = Environment.GetEnvironmentVariable("DESIREEIA_VERBOSE") == "1"
        ? (msg => Console.Error.WriteLine($"[native] {msg}"))
        : null;
    var model = LocalModel.Load(modelPath, plan, logger);

    // Recover-LoRA: --lora/--lora-scale take priority over the
    // DESIREEIA_LORA_PATH/DESIREEIA_LORA_SCALE env var defaults, mirroring
    // how every other tier/quant knob here is parametrized (CLI flag first,
    // env var fallback, hardcoded nothing).
    var loraPath = (args is not null ? GetStringOption(args, "--lora", null) : null)
        ?? Environment.GetEnvironmentVariable("DESIREEIA_LORA_PATH");
    if (!string.IsNullOrEmpty(loraPath))
    {
        var loraScaleStr = (args is not null ? GetStringOption(args, "--lora-scale", null) : null)
            ?? Environment.GetEnvironmentVariable("DESIREEIA_LORA_SCALE");
        var loraScale = loraScaleStr is not null &&
                         float.TryParse(loraScaleStr, System.Globalization.NumberStyles.Float,
                                         System.Globalization.CultureInfo.InvariantCulture, out var s)
            ? s : 1.0f;
        model.LoadLoraAdapter(loraPath, loraScale);
    }

    // Prerouter: same CLI-flag-first, env-var-fallback pattern as --lora.
    var prerouterPath = (args is not null ? GetStringOption(args, "--prerouter", null) : null)
        ?? Environment.GetEnvironmentVariable("DESIREEIA_PREROUTER_PATH");
    if (!string.IsNullOrEmpty(prerouterPath))
    {
        model.LoadPrerouter(prerouterPath);
    }
    var heuristicFlag = args is not null && HasFlag(args, "--prerouter-heuristic");
    var heuristicEnv = Environment.GetEnvironmentVariable("DESIREEIA_PREROUTER_HEURISTIC") == "1";
    if (heuristicFlag || heuristicEnv)
    {
        model.SetPrerouterHeuristic(true);
    }

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

static async Task<int> CmdGenerate(string[] rest)
{
    if (rest.Length < 2)
    {
        Console.Error.WriteLine("uso: desireeia-cli generate <modello.gguf> \"<testo>\" [--max-tokens N] [--chat] " +
                                 "[--stop <s>]... [--json [--schema <json-schema>]]");
        return 1;
    }

    var maxTokens = GetIntOption(rest, "--max-tokens", 32);
    var useChat = HasFlag(rest, "--chat");
    var stopSequences = GetStringListOption(rest, "--stop");
    var jsonMode = HasFlag(rest, "--json");
    var jsonSchema = GetStringOption(rest, "--schema", null);
    var (model, _) = Open(rest[0], args: rest);
    using (model)
    {
        ApplySamplingOptions(model, rest);
        var userText = jsonMode ? rest[1] + "\n\n" + StructuredOutput.BuildJsonInstruction(jsonSchema) : rest[1];
        var promptText = useChat
            ? model.ApplyChatTemplate(new[] { ("user", userText) })
            : userText;
        var ids = model.Tokenize(promptText);
        if (ids is null || ids.Length == 0)
        {
            Console.Error.WriteLine("tokenizzazione fallita o prompt vuoto (tokenizer non riconosciuto?).");
            return 1;
        }

        Console.Write(rest[1]);
        var options = new GenerateOptions { MaxTokens = maxTokens, StopSequences = stopSequences };
        var sb = new System.Text.StringBuilder();
        await foreach (var piece in model.StreamAsync(ids, options))
        {
            Console.Write(piece);
            sb.Append(piece);
        }
        Console.WriteLine();
        Console.WriteLine();
        if (stopSequences is { Count: > 0 })
        {
            Console.WriteLine($"[stop sequences attive: {string.Join(", ", stopSequences.Select(s => $"\"{s}\""))} - non incluse nell'output se raggiunte]");
        }
        if (jsonMode)
        {
            var extracted = StructuredOutput.TryExtractJson(sb.ToString());
            Console.WriteLine(extracted is not null
                ? $"[json mode: blocco JSON estratto correttamente ({extracted.Length} caratteri)]"
                : "[json mode: NESSUN blocco JSON valido trovato nella risposta - il modello non ha seguito l'istruzione]");
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

    var (model, plan) = Open(rest[0], threadsOpt > 0 ? threadsOpt : null, rest);
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

static int CmdChat(string[] rest)
{
    if (rest.Length < 1)
    {
        Console.Error.WriteLine("usage: desireeia-cli chat <model.gguf> [--temp T] [--top-k K] [--top-p P] [--max-tokens N] [--lora <adapter.gguf>] [--lora-scale S]");
        return 1;
    }

    var maxTokens = GetIntOption(rest, "--max-tokens", 512);
    var (model, _) = Open(rest[0], args: rest);
    using (model)
    {
        ApplySamplingOptions(model, rest);

        Console.Write("System prompt (press Enter to skip): ");
        var systemPrompt = Console.ReadLine()?.Trim() ?? "";
        var hasSystem = !string.IsNullOrEmpty(systemPrompt);

        var history = new List<(string Role, string Content)>();
        var lastAssistantResponse = "";
        if (hasSystem)
        {
            history.Add(("system", systemPrompt));
        }

        Console.WriteLine();
        Console.WriteLine("Chat started. Commands:");
        Console.WriteLine("  /exit, /quit   - quit chat");
        Console.WriteLine("  /clear         - reset conversation");
        Console.WriteLine("  /image <path> [question] - attach an image to the next reply");
        Console.WriteLine("                  (encoded via the model's vision encoder when");
        Console.WriteLine("                  available, base64 text otherwise)");
        Console.WriteLine("  /save <path>   - save last assistant response to file");
        Console.WriteLine("  /saveb64 <path>- decode last base64 block and save as binary file");
        Console.WriteLine();

        // Multimodal state: set by /image for vision models, consumed by the
        // next turn (PredictWithImage). Cleared after the turn completes so
        // later turns never re-tokenize dangling placeholders.
        float[]? pendingVisionEmbd = null;
        string pendingImageText = "";
        int pendingVisionIndex = -1;

        while (true)
        {
            Console.Write("you> ");
            var input = Console.ReadLine();
            if (input is null) break;
            input = input.Trim();
            if (input.Length == 0) continue;

            var cmd = input.ToLowerInvariant();
            if (cmd is "/exit" or "/quit") break;
            if (cmd is "/clear")
            {
                history.Clear();
                if (hasSystem) history.Add(("system", systemPrompt));
                Console.WriteLine("[conversation cleared]");
                Console.WriteLine();
                continue;
            }
            if (cmd.StartsWith("/image "))
            {
                var rest2 = input.Substring(7).Trim();
                var sp = rest2.IndexOfAny(new[] { ' ', '\t' });
                var imgPath = sp < 0 ? rest2 : rest2.Substring(0, sp);
                var question = sp < 0 ? "" : rest2.Substring(sp + 1).Trim();

                if (model.HasVision)
                {
                    // Real multimodal path: encode the image through the
                    // model's vision encoder and inject the embeddings at the
                    // placeholder tokens. history gets a message made of
                    // "<image>" repeated once per vision embedding; the turn
                    // block that follows will call PredictWithImage.
                    try
                    {
                        using var img = LocalModel.LoadImage(imgPath, 3);
                        var embd = model.EncodeImage(img, out var embdDim);
                        if (embd is null || embd.Length == 0)
                        {
                            Console.Error.WriteLine("[vision encode failed: no encoder or empty output]");
                        }
                        else
                        {
                            var nTok = Math.Max(model.VisionTokenCount, 1);
                            var userMsg = string.Concat(Enumerable.Repeat("<image>", nTok));
                            if (question.Length > 0) userMsg += "\n" + question;
                            history.Add(("user", userMsg));

                            pendingVisionEmbd = embd;
                            pendingImageText = question;
                            pendingVisionIndex = history.Count - 1;

                            Console.WriteLine($"[image {Path.GetFileName(imgPath)} encoded -> {nTok} vision tokens x {embdDim} dims]");
                            // Fall through: this turn is the image turn.
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"[error encoding image: {ex.Message}]");
                    }
                    if (pendingVisionEmbd is null)
                    {
                        Console.WriteLine();
                        continue;
                    }
                }
                else
                {
                    if (!File.Exists(imgPath))
                    {
                        Console.Error.WriteLine($"[error: file not found: {imgPath}]");
                        Console.WriteLine();
                        continue;
                    }
                    var imgBytes = File.ReadAllBytes(imgPath);
                    var b64 = Convert.ToBase64String(imgBytes);
                    var ext = Path.GetExtension(imgPath).ToLowerInvariant();
                    var mime = ext switch
                    {
                        ".png"  => "image/png",
                        ".jpg"  => "image/jpeg",
                        ".jpeg" => "image/jpeg",
                        ".gif"  => "image/gif",
                        ".bmp"  => "image/bmp",
                        ".webp" => "image/webp",
                        _       => "application/octet-stream"
                    };
                    var imgMsg = $"[image file: {Path.GetFileName(imgPath)}, {imgBytes.Length} bytes, {mime}]\n[base64: {b64}]";
                    if (question.Length > 0) imgMsg += "\n" + question;
                    history.Add(("user", imgMsg));
                    Console.WriteLine($"[loaded {Path.GetFileName(imgPath)}: {imgBytes.Length} bytes, {b64.Length} chars base64]");
                    Console.WriteLine();
                    continue;
                }
            }
            if (cmd.StartsWith("/save "))
            {
                var savePath = input.Substring(6).Trim();
                if (string.IsNullOrEmpty(savePath))
                {
                    Console.WriteLine("[usage: /save <path>]");
                }
                else if (string.IsNullOrEmpty(lastAssistantResponse))
                {
                    Console.WriteLine("[no assistant response to save]");
                }
                else
                {
                    try
                    {
                        File.WriteAllText(savePath, lastAssistantResponse);
                        Console.WriteLine($"[saved {lastAssistantResponse.Length} chars to {savePath}]");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[error saving file: {ex.Message}]");
                    }
                }
                Console.WriteLine();
                continue;
            }
            if (cmd.StartsWith("/saveb64 "))
            {
                var savePath = input.Substring(9).Trim();
                if (string.IsNullOrEmpty(savePath))
                {
                    Console.WriteLine("[usage: /saveb64 <path>]");
                }
                else if (string.IsNullOrEmpty(lastAssistantResponse))
                {
                    Console.WriteLine("[no assistant response to decode]");
                }
                else
                {
                    try
                    {
                        var b64Start = lastAssistantResponse.LastIndexOf("base64: ");
                        if (b64Start < 0)
                        {
                            Console.WriteLine("[no base64 block found in last response]");
                        }
                        else
                        {
                            b64Start += 8;
                            var b64End = lastAssistantResponse.IndexOf(']', b64Start);
                            if (b64End < 0) b64End = lastAssistantResponse.Length;
                            var b64Data = lastAssistantResponse.Substring(b64Start, b64End - b64Start).Trim();
                            var bytes = Convert.FromBase64String(b64Data);
                            File.WriteAllBytes(savePath, bytes);
                            Console.WriteLine($"[saved {bytes.Length} bytes to {savePath}]");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[error saving file: {ex.Message}]");
                    }
                }
                Console.WriteLine();
                continue;
            }

            bool imageTurn = pendingVisionEmbd != null;
            if (!imageTurn) history.Add(("user", input));

            var messages = history.ToList();
            var promptText = model.ApplyChatTemplate(messages, addAssistant: true);
            // addBos left at its default (true): Predict() resets the KV
            // cache and re-prefills the whole reconstructed history text on
            // every turn (see engine_predict's reset_cache()), so this is a
            // fresh sequence start each time, not a continuation — the same
            // situation "generate --chat" is already in, and that path has
            // always left addBos at its default. Explicitly turning it off
            // here meant every turn's prompt started without the sequence-
            // start marker the model was trained to expect from token 0,
            // which is why the observed output degenerated into unrelated
            // fragments immediately.
            var ids = model.Tokenize(promptText);
            if (ids is null || ids.Length == 0)
            {
                Console.Error.WriteLine("tokenization failed.");
                if (imageTurn && pendingVisionIndex >= 0) history.RemoveAt(pendingVisionIndex);
                else history.RemoveAt(history.Count - 1);
                pendingVisionEmbd = null;
                pendingVisionIndex = -1;
                Console.WriteLine();
                continue;
            }

            Console.Write("assistant> ");
            var sb = new System.Text.StringBuilder();
            int next;
            try
            {
                next = imageTurn
                    ? model.PredictWithImage(ids, pendingVisionEmbd!, imageToken: -1)
                    : model.Predict(ids);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[error: {ex.Message}]");
                if (pendingVisionIndex >= 0) history.RemoveAt(pendingVisionIndex);
                pendingVisionEmbd = null;
                pendingVisionIndex = -1;
                Console.WriteLine();
                continue;
            }
            var stopped = model.IsEndOfGeneration(next);
            if (!stopped)
            {
                var piece = model.TokenPiece(next) ?? "";
                sb.Append(piece);
                Console.Write(piece);
                for (int i = 1; i < maxTokens; i++)
                {
                    next = model.NextToken();
                    if (model.IsEndOfGeneration(next)) { stopped = true; break; }
                    piece = model.TokenPiece(next) ?? "";
                    sb.Append(piece);
                    Console.Write(piece);
                }
            }
            Console.WriteLine();
            Console.WriteLine();

            lastAssistantResponse = sb.ToString();
            history.Add(("assistant", lastAssistantResponse));

            // Multimodal cleanup: replace the "<image>..." placeholder message
            // with a short text remnant so subsequent turns never re-tokenize
            // dangling placeholders without their embeddings, and clear the
            // pending vision state.
            if (imageTurn && pendingVisionIndex >= 0)
            {
                pendingVisionEmbd = null;
                var remnant = "[image attached]";
                if (pendingImageText.Length > 0) remnant += " " + pendingImageText;
                history[pendingVisionIndex] = ("user", remnant);
                pendingVisionIndex = -1;
                pendingImageText = "";
            }
        }
    }
    return 0;
}

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

// Raccoglie TUTTE le occorrenze di un'opzione ripetibile (es. piu' --stop),
// a differenza di GetStringOption che si ferma alla prima. Null se assente.
static List<string>? GetStringListOption(string[] args, string name)
{
    List<string>? result = null;
    for (int i = 0; i < args.Length - 1; i++)
    {
        if (args[i] == name)
        {
            (result ??= new List<string>()).Add(args[i + 1]);
        }
    }
    return result;
}
