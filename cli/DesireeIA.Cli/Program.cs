// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

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
        "autoconfig" => CmdAutoConfig(rest),
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
    Console.Error.WriteLine($"native library not found: {ex.Message}");
    Console.Error.WriteLine("build the native engine first (cmake + C++17 compiler).");
    return 2;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

static int Unknown(string cmd)
{
    Console.Error.WriteLine($"unknown command: {cmd}");
    PrintUsage();
    return 1;
}

static void PrintUsage()
{
    Console.WriteLine("""
        DesireeIA CLI - test client for the native engine

        Usage:
          desireeia-cli hw
          desireeia-cli info <model.gguf>
          desireeia-cli autoconfig <model.gguf>  (parameters chosen for this machine and model)
          desireeia-cli tokenize <model.gguf> "<text>"
          desireeia-cli embed <model.gguf> "<text>"  (BERT encoder only)
          desireeia-cli generate <model.gguf> "<text>" [--max-tokens N] [--chat]
                     [--temp T] [--top-k K] [--top-p P] [--greedy]
                     [--repeat-penalty R] [--repeat-last-n N] [--seed S]
                     [--stop "<s>"]...  (repeatable: stops generation at the first occurrence,
                                         excluded from the output - see GenerateOptions.StopSequences)
                     [--json [--schema "<json-schema>"]]  (best-effort: instructs the model to
                                         respond with JSON only and extracts the valid block;
                                         not grammar-constrained decoding)
                     [--lora <adapter.gguf>] [--lora-scale S]  (Recover-LoRA, see README)
                     [--prerouter <file.gguf>] [--prerouter-heuristic]  (see README)
          desireeia-cli bench <model.gguf> [--tokens N] [--warmup N] [--prompt "<text>"] [--threads N]
                     [--lora <adapter.gguf>] [--lora-scale S]
                     [--prerouter <file.gguf>] [--prerouter-heuristic]
          desireeia-cli chat <model.gguf> [--temp T] [--top-k K] [--top-p P] [--max-tokens N]
                     [--lora <adapter.gguf>] [--lora-scale S]
                     [--prerouter <file.gguf>] [--prerouter-heuristic]

        Chat session:
          desireeia-cli chat <model.gguf>  (interactive multi-turn conversation)
          /image <path>   load image file and inject into context (base64)
          /save <path>    save last assistant response to file
          /saveb64 <path> decode last base64 block and save as binary file

        Sampling: without options the automatic defaults are used
        (--temp 0.7 --top-k 40 --top-p 0.9, see `autoconfig`); --greedy
        switches to deterministic argmax decoding, which tends to fall into
        repetitive loops on long generations. The backend, threads and RAM
        budget are always picked for this machine (--backend cpu|cuda forces one).
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
        throw new FileNotFoundException($"model not found: {modelPath}");
    }
    // --backend forces the backend instead of letting it auto-detect
    // (ExecutionPlan.Backend defaults to Unconfigured precisely so it
    // doesn't interfere when not explicitly requested): used to compare
    // CPU and CUDA on the same model/hardware in the benchmark.
    var backendOpt = args is not null ? GetStringOption(args, "--backend", null) : null;
    InferenceBackend? backend = backendOpt?.ToLowerInvariant() switch
    {
        null => null,
        "cpu" => InferenceBackend.Cpu,
        "cuda" => InferenceBackend.Cuda,
        _ => throw new ArgumentException($"unknown --backend: {backendOpt} (use cpu or cuda)")
    };
    ExecutionPlan? overrides = null;
    if (threads is int t && backend is InferenceBackend b) overrides = new ExecutionPlan { ThreadCount = t, Backend = b };
    else if (threads is int t2) overrides = new ExecutionPlan { ThreadCount = t2 };
    else if (backend is InferenceBackend b2) overrides = new ExecutionPlan { Backend = b2 };
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

static int CmdAutoConfig(string[] rest)
{
    if (rest.Length < 1)
    {
        Console.Error.WriteLine("usage: desireeia-cli autoconfig <model.gguf>");
        return 1;
    }
    var c = AutoConfigurator.Configure(rest[0]);
    var m = c.Model;
    Console.WriteLine($"model       = {Path.GetFileName(rest[0])} ({m.Architecture}, {m.LayerCount} layers, " +
                      $"trained context {m.TrainedContextLength}, {m.FileSizeBytes / (1024 * 1024)} MB)");
    Console.WriteLine($"plan        = {c.Plan}");
    Console.WriteLine($"sampling    = --temp {c.Sampling.Temperature} --top-k {c.Sampling.TopK} --top-p {c.Sampling.TopP}");
    Console.WriteLine($"context     = {c.ContextSize}");
    Console.WriteLine($"max-tokens  = {c.MaxTokens}");
    foreach (var note in c.Notes) Console.WriteLine($"  - {note}");
    return 0;
}

static int CmdInfo(string[] rest)
{
    if (rest.Length < 1)
    {
        Console.Error.WriteLine("usage: desireeia-cli info <model.gguf>");
        return 1;
    }

    var (model, plan) = Open(rest[0]);
    using (model)
    {
        Console.WriteLine($"path        = {model.ModelPath}");
        Console.WriteLine($"plan        = {plan}");
        Console.WriteLine($"tokenizer   = {(model.HasTokenizer ? "recognized (SPM or BPE)" : "not recognized")}");
        Console.WriteLine($"context_size = {model.ContextSize()} bytes (0 before the first Predict)");
    }
    return 0;
}

static int CmdTokenize(string[] rest)
{
    if (rest.Length < 2)
    {
        Console.Error.WriteLine("usage: desireeia-cli tokenize <model.gguf> \"<text>\"");
        return 1;
    }

    var (model, _) = Open(rest[0]);
    using (model)
    {
        var ids = model.Tokenize(rest[1]);
        if (ids is null)
        {
            Console.Error.WriteLine("the model does not have a recognized tokenizer (neither SentencePiece nor BPE).");
            return 1;
        }
        foreach (var id in ids)
        {
            var piece = model.TokenPiece(id) ?? "?";
            Console.WriteLine($"{id}\t{piece}");
        }
        Console.WriteLine($"({ids.Length} tokens)");
    }
    return 0;
}

static int CmdEmbed(string[] rest)
{
    if (rest.Length < 2)
    {
        Console.Error.WriteLine("usage: desireeia-cli embed <model.gguf> \"<text>\"");
        return 1;
    }

    var (model, _) = Open(rest[0]);
    using (model)
    {
        var ids = model.Tokenize(rest[1]);
        if (ids is null)
        {
            Console.Error.WriteLine("the model does not have a recognized tokenizer.");
            return 1;
        }
        var rows = model.Embed(ids);
        if (rows is null)
        {
            Console.Error.WriteLine("the loaded model is not a BERT encoder (desireeia_embed not supported).");
            return 1;
        }
        Console.WriteLine($"{ids.Length} tokens, embedding dimension = {(rows.Length > 0 ? rows[0].Length : 0)}");

        if (rows.Length == 0) return 0;
        var dim = rows[0].Length;
        var pooled = new float[dim];
        foreach (var row in rows)
            for (int i = 0; i < dim; i++) pooled[i] += row[i] / rows.Length;
        var norm = MathF.Sqrt(pooled.Sum(v => v * v));
        Console.WriteLine($"mean-pool (first 8 components, L2 norm={norm:F4}): " +
            string.Join(", ", pooled.Take(8).Select(v => v.ToString("F4"))));
    }
    return 0;
}

static async Task<int> CmdGenerate(string[] rest)
{
    if (rest.Length < 2)
    {
        Console.Error.WriteLine("usage: desireeia-cli generate <model.gguf> \"<text>\" [--max-tokens N] [--chat] " +
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
            Console.Error.WriteLine("tokenization failed or empty prompt (unrecognized tokenizer?).");
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
            Console.WriteLine($"[active stop sequences: {string.Join(", ", stopSequences.Select(s => $"\"{s}\""))} - excluded from output if reached]");
        }
        if (jsonMode)
        {
            var extracted = StructuredOutput.TryExtractJson(sb.ToString());
            Console.WriteLine(extracted is not null
                ? $"[json mode: JSON block successfully extracted ({extracted.Length} characters)]"
                : "[json mode: NO valid JSON block found in the response - the model did not follow the instruction]");
        }
        var s = model.GetSampling();
        Console.WriteLine(s.Temperature > 0
            ? $"[sampling: temp={s.Temperature} top-k={s.TopK} top-p={s.TopP} " +
              $"repeat-penalty={s.PenaltyRepeat} (last {s.PenaltyLastN})]"
            : "[greedy (argmax, --greedy): deterministic. On long generations it falls into " +
              "repetitive loops: drop --greedy to use the automatic sampling]");
        Console.WriteLine("[NB: no logit-by-logit correctness validation against a " +
                           "reference - see docs/engine_gap_analysis.md]");
    }
    return 0;
}

static int CmdBench(string[] rest)
{
    if (rest.Length < 1)
    {
        Console.Error.WriteLine("usage: desireeia-cli bench <model.gguf> [--tokens N] [--warmup N] [--prompt \"<text>\"] [--threads N] [--backend cpu|cuda]");
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
        Console.WriteLine($"profile (decode only, {nTokens} tokens): {DesireeIAEngine.ProfileDump()}");
        Console.WriteLine($"context_size after the bench: {model.ContextSize()} bytes");
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

    var maxTokens = GetIntOption(rest, "--max-tokens",
        AutoConfigurator.Compute(DesireeIAEngine.DetectHardware(), DesireeIAEngine.BuildPlan(rest[0]),
                                 ModelTraits.Read(rest[0])).MaxTokens);
    var (model, plan) = Open(rest[0], args: rest);
    using (model)
    {
        ApplySamplingOptions(model, rest);

        // Which model and which backend are actually active, before the
        // first prompt: with auto-detection the answer isn't obvious from
        // the command line, and knowing whether a run went to the GPU or
        // quietly fell back to CPU is the first thing worth seeing.
        Console.WriteLine();
        Console.WriteLine($"model    : {Path.GetFileName(rest[0])}");
        Console.WriteLine($"backend  : {plan.Backend}   threads: {plan.ThreadCount}   dense: {plan.DenseQuantization}   max-tokens: {maxTokens}");

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
            // Timed separately from the decode loop: the first token also
            // pays for the prompt (prefill), so folding it into the
            // per-token rate would understate the real decode speed.
            var swPrefill = System.Diagnostics.Stopwatch.StartNew();
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
            swPrefill.Stop();
            var stopped = model.IsEndOfGeneration(next);
            var swDecode = System.Diagnostics.Stopwatch.StartNew();
            int generated = 0;
            if (!stopped)
            {
                var piece = model.TokenPiece(next) ?? "";
                sb.Append(piece);
                Console.Write(piece);
                generated = 1;
                for (int i = 1; i < maxTokens; i++)
                {
                    next = model.NextToken();
                    if (model.IsEndOfGeneration(next)) { stopped = true; break; }
                    piece = model.TokenPiece(next) ?? "";
                    sb.Append(piece);
                    Console.Write(piece);
                    generated++;
                }
            }
            swDecode.Stop();
            Console.WriteLine();

            // Per-turn stats: tokens produced, decode rate, and how long the
            // prompt itself took before the first token appeared.
            if (generated > 1)
            {
                var decodeMs = swDecode.Elapsed.TotalMilliseconds;
                var rate = decodeMs > 0 ? (generated - 1) * 1000.0 / decodeMs : 0.0;
                Console.WriteLine($"[{generated} tokens | {rate:F1} tok/s | prefill {swPrefill.Elapsed.TotalMilliseconds:F0} ms | {plan.Backend}]");
            }
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
    // Defaults = AutoConfigurator's (never greedy: greedy loops on long
    // generations). --greedy restores the deterministic argmax decoding.
    var temp = HasFlag(args, "--greedy") ? 0.0f
        : GetFloatOption(args, "--temp", AutoConfigurator.DefaultTemperature);
    var options = new SamplingOptions
    {
        Temperature      = temp,
        TopK             = GetIntOption(args, "--top-k", AutoConfigurator.DefaultTopK),
        TopP             = GetFloatOption(args, "--top-p", AutoConfigurator.DefaultTopP),
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

// Collects ALL occurrences of a repeatable option (e.g. multiple --stop),
// unlike GetStringOption which stops at the first one. Null if absent.
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
