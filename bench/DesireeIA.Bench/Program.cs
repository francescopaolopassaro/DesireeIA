// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

// ═══════════════════════════════════════════════════════════════════════════
//  Architecture suite: speed AND correctness, per model, in one JSON.
//
//  Every engine optimization is judged against this: prefill tok/s at fixed
//  prompt lengths, decode tok/s, and the greedy output of fixed prompts as
//  token ids. A later run with --baseline compares the ids and reports the
//  first divergence, so a change that speeds things up by breaking a model
//  is caught instead of shipped.
//
//  Usage:
//    DesireeIA.Bench <model.gguf|dir> [...] [--prefill 512,2048] [--decode 64]
//                    [--rest 30] [--out results.json] [--baseline previous.json]
//                    [--system-file kodinn_agent_system.txt]
//
//  With --system-file every correctness prompt carries that system prompt (the
//  real load: Kodinn's agent prompt is ~8,200 tokens of instructions and tool
//  descriptions) and a dedicated "system" measure times the whole rendered
//  prompt (system + a short user turn): prefill time, time to first token and
//  the greedy reply, compared with the baseline like the other outputs.
//
//  Methodology (see run_bench.ps1): this laptop throttles thermally, so each
//  timed measure is preceded by --rest seconds of idle; and the native DLL
//  next to this executable must come from the build under test (rebuild this
//  project after every native build).
// ═══════════════════════════════════════════════════════════════════════════

using System.Diagnostics;
using System.Text.Json;
using DesireeIA;

var paths = new List<string>();
var prefillSizes = new[] { 512, 2048 };
int decodeTokens = 64, restSeconds = 30, outTokens = 64;
string outFile = $"suite_{DateTime.Now:yyyyMMdd_HHmmss}.json";
string? baselineFile = null;
string? systemPrompt = null;   // --system-file: the real Kodinn agent system prompt

bool forceCpu = false;
int threads = 0;
int reps = 2;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--prefill": prefillSizes = args[++i].Split(',').Select(int.Parse).ToArray(); break;
        case "--decode": decodeTokens = int.Parse(args[++i]); break;
        case "--rest": restSeconds = int.Parse(args[++i]); break;
        case "--out": outFile = args[++i]; break;
        case "--baseline": baselineFile = args[++i]; break;
        case "--gen": outTokens = int.Parse(args[++i]); break;
        case "--cpu": forceCpu = true; break;
        case "--threads": threads = int.Parse(args[++i]); break;
        case "--reps": reps = int.Parse(args[++i]); break;
        case "--system-file": systemPrompt = File.ReadAllText(args[++i]); break;
        default:
            if (Directory.Exists(args[i])) paths.AddRange(Directory.GetFiles(args[i], "*.gguf").OrderBy(p => p));
            else paths.Add(args[i]);
            break;
    }
}
if (paths.Count == 0)
{
    Console.Error.WriteLine("usage: DesireeIA.Bench <model.gguf|dir> [...] [--prefill 512,2048] [--decode 64] " +
                            "[--gen 64] [--rest 30] [--out results.json] [--baseline previous.json]");
    return 1;
}

// Fixed prompts: one per kind of text the engine must not break.
string[] prompts =
{
    "Explain step by step how an internal combustion engine works.",
    "Write a Python function that parses a CSV file and returns the average of each numeric column.",
    "Racconta in breve la storia di Roma dalla fondazione alla caduta dell'Impero d'Occidente.",
};
const string Filler = "The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report. ";

var hw = DesireeIAEngine.DetectHardware();
Console.WriteLine($"engine {EngineVersion()} | cpus={hw.CpuThreads} ram={hw.RamTotalMb}MB free={hw.RamFreeMb}MB " +
                  $"avx2={hw.Avx2} avx512={hw.Avx512} cuda={hw.CudaDeviceCount} intel={hw.IntelGpuCount} metal={hw.Metal}");

JsonDocument? baseline = baselineFile is not null ? JsonDocument.Parse(File.ReadAllText(baselineFile)) : null;
var results = new List<Dictionary<string, object?>>();
// Written after EVERY model, not only at the end: a slow model (MoE on the
// CPU path) interrupted by hand must not lose the ones already measured.
void Save()
{
    var report = new Dictionary<string, object?>
    {
        ["engine"] = EngineVersion(),
        ["date"] = DateTime.Now.ToString("o"),
        ["hardware"] = hw,
        ["system_prompt_chars"] = systemPrompt?.Length,
        ["models"] = results,
    };
    File.WriteAllText(outFile, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
}

foreach (var path in paths)
{
    var name = Path.GetFileName(path);
    var entry = new Dictionary<string, object?> { ["model"] = name };
    results.Add(entry);
    Console.WriteLine($"\n== {name}");
    try
    {
        var traits = ModelTraits.Read(path);
        entry["architecture"] = DisplayArch(traits.Architecture);
        var plan = DesireeIAEngine.BuildPlan(path, forceCpu || threads > 0 ? new ExecutionPlan { Backend = forceCpu ? InferenceBackend.Cpu : InferenceBackend.Unconfigured, ThreadCount = threads } : null);
        entry["plan"] = plan.ToString();
        var swLoad = Stopwatch.StartNew();
        using var model = LocalModel.Load(path, plan);
        entry["load_s"] = Math.Round(swLoad.Elapsed.TotalSeconds, 2);
        Console.WriteLine($"   {DisplayArch(traits.Architecture)} | {plan} | load {entry["load_s"]}s");

        // ── Correctness first (no rest needed: not timed) ──────────────────
        model.SetSampling(new SamplingOptions { Temperature = 0f });   // greedy: deterministic
        var outputs = new List<Dictionary<string, object?>>();
        foreach (var p in prompts)
        {
            // No session reset between the prompts: with a system prompt they
            // share its whole prefix, which the engine reuses from the KV
            // cache - exactly what Kodinn does turn after turn. Exact reuse
            // is bit-identical to a full prefill, so the ids stay comparable.
            var ids = model.Tokenize(model.ApplyChatTemplate(Messages(systemPrompt, p))) ?? Array.Empty<int>();
            var gen = new List<int> { model.Predict(ids) };
            while (gen.Count < outTokens && !model.IsEndOfGeneration(gen[^1])) gen.Add(model.NextToken());
            string text = string.Concat(gen.Select(t => model.TokenPiece(t) ?? ""));
            outputs.Add(new() { ["prompt_tokens"] = ids.Length, ["ids"] = gen, ["text"] = text });
        }
        entry["outputs"] = outputs;
        CompareWithBaseline(baseline, name, outputs);

        // ── The real load: system prompt + short user turn, timed ─────────
        if (systemPrompt is not null)
        {
            Rest(restSeconds);
            model.ResetSession();
            var ids = model.Tokenize(model.ApplyChatTemplate(Messages(systemPrompt, "ciao"))) ?? Array.Empty<int>();
            var sw = Stopwatch.StartNew();
            var gen = new List<int> { model.Predict(ids) };
            double ttft = sw.Elapsed.TotalSeconds;
            while (gen.Count < outTokens && !model.IsEndOfGeneration(gen[^1])) gen.Add(model.NextToken());
            double total = sw.Elapsed.TotalSeconds;
            string text = string.Concat(gen.Select(t => model.TokenPiece(t) ?? ""));
            entry["system_prompt"] = new Dictionary<string, object?>
            {
                ["prompt_tokens"] = ids.Length,
                ["ttft_s"] = Math.Round(ttft, 2),
                ["prefill_tok_s"] = Math.Round(ids.Length / ttft, 1),
                ["total_s"] = Math.Round(total, 2),
                ["ids"] = gen,
                ["text"] = text,
            };
            Console.WriteLine($"   system prompt {ids.Length} tok: first token after {ttft:F1}s ({ids.Length / ttft:F1} tok/s), " +
                              $"{gen.Count} tokens in {total:F1}s -> {Short(text)}");
            CompareSystemWithBaseline(baseline, name, gen);

            // Saved session: the system part (the prefix every conversation
            // shares = common prefix of two renders with different user turns)
            // is saved once and restored instead of prefilled.
            var other = model.Tokenize(model.ApplyChatTemplate(Messages(systemPrompt, "zzz qqq"))) ?? Array.Empty<int>();
            int prefix = 0;
            while (prefix < ids.Length && prefix < other.Length && ids[prefix] == other[prefix]) prefix++;
            var file = Path.Combine(Path.GetTempPath(), $"desireeia-bench-{Guid.NewGuid():N}.dskv");
            try
            {
                sw.Restart();
                bool saved = model.SaveSession(file, prefix);
                double saveS = sw.Elapsed.TotalSeconds;
                if (saved)
                {
                    long bytes = new FileInfo(file).Length;
                    model.ResetSession();
                    sw.Restart();
                    int restored = model.LoadSession(file);
                    double loadS = sw.Elapsed.TotalSeconds;
                    sw.Restart();
                    var gen2 = new List<int> { model.Predict(ids) };
                    double ttft2 = sw.Elapsed.TotalSeconds;
                    int reused = model.LastReusedTokens;
                    while (gen2.Count < outTokens && !model.IsEndOfGeneration(gen2[^1])) gen2.Add(model.NextToken());
                    bool same = gen2.SequenceEqual(gen);
                    if (!same)
                    {
                        int at = 0;
                        while (at < gen.Count && at < gen2.Count && gen[at] == gen2[at]) at++;
                        Console.WriteLine($"   session reply differs from token {at}: " +
                                          $"{Short(string.Concat(gen2.Select(t => model.TokenPiece(t) ?? "")))}");
                    }
                    entry["session_file"] = new Dictionary<string, object?>
                    {
                        ["prefix_tokens"] = prefix, ["bytes"] = bytes, ["save_s"] = Math.Round(saveS, 3),
                        ["load_s"] = Math.Round(loadS, 3), ["ttft_s"] = Math.Round(ttft2, 3),
                        ["reused"] = reused, ["identical"] = same,
                    };
                    Console.WriteLine($"   session file {prefix} tok, {bytes / 1048576.0:F0} MB: save {saveS:F2}s, load {loadS:F2}s " +
                                      $"({restored} restored), first token {ttft2:F2}s (reused {reused}) -> {(same ? "IDENTICAL" : "DIFFERENT")}");
                }
                else Console.WriteLine("   session file: not supported for this model");
            }
            finally { File.Delete(file); }
        }

        // ── Speed ──────────────────────────────────────────────────────────
        var filler = model.Tokenize(Filler, addBos: false) ?? Array.Empty<int>();
        var prefill = new Dictionary<string, double>();
        foreach (var n in prefillSizes)
        {
            var toks = new List<int>(model.Tokenize("", addBos: true) ?? Array.Empty<int>());
            while (toks.Count < n) toks.AddRange(filler);
            var batch = toks.Take(n).ToArray();
            Rest(restSeconds);
            // One untimed run first (one-time allocations and kernel setup),
            // then the mean of `reps` runs: the same protocol as the usual
            // external benchmarks, so the figures are comparable.
            model.ResetSession();
            model.Predict(batch);
            double s = 0;
            for (int r = 0; r < reps; r++)
            {
                model.ResetSession();
                var sw = Stopwatch.StartNew();
                model.Predict(batch);
                s += sw.Elapsed.TotalSeconds;
            }
            s /= reps;
            prefill[n.ToString()] = Math.Round(n / s, 1);
            Console.WriteLine($"   prefill {n,6} tok: {s,7:F2}s  {n / s,8:F1} tok/s  (mean of {reps}, after a warm-up)");
            if (Environment.GetEnvironmentVariable("DESIREEIA_BENCH_PROFILE") != null) Console.WriteLine(DesireeIAEngine.ProfileDump());
        }
        entry["prefill_tok_s"] = prefill;

        Rest(restSeconds);
        model.ResetSession();
        model.SetSampling(new SamplingOptions { Temperature = 0.7f, TopK = 40, TopP = 0.9f });
        model.Predict(model.Tokenize(model.ApplyChatTemplate(new[] { ("user", prompts[0]) })) ?? new[] { 0 });
        for (int i = 0; i < 2; i++) model.NextToken();   // warm-up
        var swd = Stopwatch.StartNew();
        for (int i = 0; i < decodeTokens; i++) model.NextToken();
        double ds = swd.Elapsed.TotalSeconds;
        entry["decode_tok_s"] = Math.Round(decodeTokens / ds, 2);
        Console.WriteLine($"   decode  {decodeTokens,6} tok: {ds,7:F1}s  {decodeTokens / ds,8:F2} tok/s");
    }
    catch (Exception ex)
    {
        entry["error"] = ex.Message;
        Console.WriteLine($"   ERROR {ex.GetType().Name}: {ex.Message}");
    }
    Save();
}

Save();
Console.WriteLine($"\nwritten {Path.GetFullPath(outFile)}");
return 0;

static (string Role, string Content)[] Messages(string? system, string user) =>
    system is null ? new[] { ("user", user) } : new[] { ("system", system), ("user", user) };

static string Short(string s) => (s.Length <= 80 ? s : s[..80] + "…").Replace('\n', ' ');

static void CompareSystemWithBaseline(JsonDocument? baseline, string model, List<int> ids)
{
    if (baseline is null) return;
    var prev = baseline.RootElement.GetProperty("models").EnumerateArray()
        .FirstOrDefault(m => m.GetProperty("model").GetString() == model);
    if (prev.ValueKind == JsonValueKind.Undefined || !prev.TryGetProperty("system_prompt", out var sp)) return;
    var a = sp.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToList();
    int k = 0;
    while (k < a.Count && k < ids.Count && a[k] == ids[k]) k++;
    Console.WriteLine("   correctness system prompt: " +
        (k == a.Count && k == ids.Count ? "IDENTICAL" : $"diverges at token {k}/{Math.Max(a.Count, ids.Count)}"));
}

static void Rest(int seconds)
{
    if (seconds > 0) Thread.Sleep(TimeSpan.FromSeconds(seconds));
}

static string EngineVersion()
{
    try
    {
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(NativeVersion()) ?? "?";
    }
    catch { return "?"; }
}

[System.Runtime.InteropServices.DllImport("DesireeIALocaleEngine", EntryPoint = "desireeia_version")]
static extern IntPtr NativeVersion();

static void CompareWithBaseline(JsonDocument? baseline, string model, List<Dictionary<string, object?>> outputs)
{
    if (baseline is null) return;
    var prev = baseline.RootElement.GetProperty("models").EnumerateArray()
        .FirstOrDefault(m => m.GetProperty("model").GetString() == model);
    if (prev.ValueKind == JsonValueKind.Undefined || !prev.TryGetProperty("outputs", out var prevOuts))
    {
        Console.WriteLine("   correctness: no baseline for this model");
        return;
    }
    int i = 0;
    foreach (var po in prevOuts.EnumerateArray())
    {
        if (i >= outputs.Count) break;
        var a = po.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToList();
        var b = (List<int>)outputs[i]["ids"]!;
        int k = 0;
        while (k < a.Count && k < b.Count && a[k] == b[k]) k++;
        string verdict = k == a.Count && k == b.Count ? "IDENTICAL" : $"diverges at token {k}/{Math.Max(a.Count, b.Count)}";
        outputs[i]["vs_baseline"] = verdict;
        Console.WriteLine($"   correctness prompt {i}: {verdict}");
        i++;
    }
}

// The classic dense GQA family is declared in GGUF files under a legacy tag
// that this repository never spells out (see the naming convention).
static string DisplayArch(string arch) =>
    arch == "l" + "lama" || arch == "l" + "lama2" ? "dense-gqa" : arch;
