using System.Text;
using System.Text.Json;

namespace DesireeIA;

/// <summary>
/// Opzioni per la generazione in streaming (<see cref="LocalModel.StreamAsync"/>
/// / <see cref="LocalModel.ChatStreamAsync"/>).
/// </summary>
public sealed class GenerateOptions
{
    /// <summary>Limite di token generati oltre al quale la generazione si ferma comunque.</summary>
    public int MaxTokens { get; init; } = 512;

    /// <summary>
    /// Se una qualsiasi di queste stringhe compare nel testo generato, la
    /// generazione si ferma e il testo restituito NON include la stringa
    /// di stop (coerente con la convenzione "stop" delle API tipo Ollama/OpenAI).
    /// </summary>
    public IReadOnlyList<string>? StopSequences { get; init; }
}

/// <summary>
/// Rileva incrementalmente, pezzo di testo dopo pezzo, se e' comparsa una
/// stop sequence. Serve perche' lo streaming produce il testo un token (o
/// frammento di token) alla volta, ma una stop sequence puo' estendersi su
/// piu' pezzi consecutivi: bisogna trattenere una piccola coda di testo non
/// ancora emesso finche' non e' certo che non faccia parte di un match.
/// </summary>
public sealed class StopSequenceScanner
{
    private readonly string[] _stops;
    private readonly int _maxStopLen;
    private readonly StringBuilder _tail = new();

    public StopSequenceScanner(IReadOnlyList<string>? stopSequences)
    {
        _stops = (stopSequences ?? Array.Empty<string>())
            .Where(s => !string.IsNullOrEmpty(s))
            .ToArray();
        _maxStopLen = _stops.Length == 0 ? 0 : _stops.Max(s => s.Length);
    }

    public bool HasStops => _stops.Length > 0;

    /// <summary>
    /// Accetta il prossimo pezzo di testo decodificato. Restituisce il testo
    /// ormai sicuro da emettere (puo' essere vuoto) e se e' stata trovata una
    /// stop sequence (nel qual caso il chiamante deve interrompere la
    /// generazione: `emit` e' tutto e solo il testo PRIMA della stop sequence).
    /// </summary>
    public (string emit, bool stopped) Feed(string piece)
    {
        if (_stops.Length == 0) return (piece, false);

        _tail.Append(piece);
        var combined = _tail.ToString();

        foreach (var stop in _stops)
        {
            var idx = combined.IndexOf(stop, StringComparison.Ordinal);
            if (idx >= 0)
            {
                var before = combined[..idx];
                _tail.Clear();
                return (before, true);
            }
        }

        // Nessun match ancora: si puo' emettere tutto tranne una coda lunga
        // quanto la piu' lunga stop sequence meno 1 carattere, perche' quella
        // coda potrebbe essere l'inizio di un match che si completa con il
        // pezzo successivo.
        var safeLen = Math.Max(0, combined.Length - (_maxStopLen - 1));
        var toEmit = combined[..safeLen];
        _tail.Clear();
        _tail.Append(combined[safeLen..]);
        return (toEmit, false);
    }

    /// <summary>
    /// Da chiamare quando la generazione finisce normalmente (EOS o limite
    /// di token) senza mai aver trovato una stop sequence: il testo rimasto
    /// in coda non ne fa parte, quindi va comunque emesso.
    /// </summary>
    public string Flush()
    {
        var s = _tail.ToString();
        _tail.Clear();
        return s;
    }
}

/// <summary>Descrizione di un tool invocabile dal modello (nome, scopo, schema JSON dei parametri).</summary>
public sealed record ToolDefinition(string Name, string Description, string ParametersJsonSchema);

/// <summary>Una richiesta di chiamata a tool estratta dal testo generato dal modello.</summary>
public sealed record ToolCall(string Name, string ArgumentsJson);

/// <summary>
/// Tool calling "prompt-based": DesireeIA non ha un formato nativo di
/// function-calling addestrato nel modello (a differenza dei modelli
/// "instruct-tools" specifici), quindi qui si chiede al modello, via
/// istruzioni di sistema, di rispondere con un blocco JSON riconoscibile
/// quando vuole invocare un tool, e lo si fa il parsing di quel blocco.
/// Funziona bene con modelli chat generici purche' seguano l'istruzione;
/// non e' affidabile quanto un grammar-constrained decoding (non implementato).
/// </summary>
public static class ToolCalling
{
    private const string OpenTag = "<tool_call>";
    private const string CloseTag = "</tool_call>";

    /// <summary>
    /// Istruzione di sistema che descrive i tool disponibili e il formato
    /// atteso per invocarli. Da anteporre (o accodare) al messaggio di
    /// sistema della conversazione.
    /// </summary>
    public static string BuildSystemPrompt(IEnumerable<ToolDefinition> tools)
    {
        var sb = new StringBuilder();
        sb.AppendLine("You can call tools to help answer the user. To call a tool, respond with ONLY this block and nothing else (no other text before or after):");
        sb.Append(OpenTag).AppendLine();
        sb.AppendLine("{\"name\": \"<tool name>\", \"arguments\": { ... }}");
        sb.Append(CloseTag).AppendLine();
        sb.AppendLine("If no tool call is needed, answer normally in plain text instead. Available tools:");
        foreach (var t in tools)
        {
            sb.AppendLine($"- {t.Name}: {t.Description}");
            sb.AppendLine($"  parameters (JSON schema): {t.ParametersJsonSchema}");
        }
        return sb.ToString();
    }

    /// <summary>
    /// Cerca un blocco &lt;tool_call&gt;...&lt;/tool_call&gt; nel testo generato
    /// e ne fa il parsing. Restituisce false se non c'e' nessun blocco o se il
    /// contenuto non e' JSON valido con un campo "name" non vuoto.
    /// </summary>
    public static bool TryParse(string responseText, out ToolCall? call)
    {
        call = null;
        if (string.IsNullOrEmpty(responseText)) return false;

        var start = responseText.IndexOf(OpenTag, StringComparison.Ordinal);
        if (start < 0) return false;
        start += OpenTag.Length;
        var end = responseText.IndexOf(CloseTag, start, StringComparison.Ordinal);
        var json = (end < 0 ? responseText[start..] : responseText[start..end]).Trim();
        if (json.Length == 0) return false;

        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (!root.TryGetProperty("name", out var nameEl)) return false;
            var name = nameEl.GetString();
            if (string.IsNullOrEmpty(name)) return false;
            var argsJson = root.TryGetProperty("arguments", out var argsEl) ? argsEl.GetRawText() : "{}";
            call = new ToolCall(name, argsJson);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>
    /// Formatta il risultato di un tool come messaggio da riaccodare alla
    /// history per il turno successivo. Usa il ruolo "user" (non "tool"):
    /// i chat template per-architettura di DesireeIA (vedi core/chat_template.cpp)
    /// riconoscono solo system/user/assistant per la maggior parte dei
    /// formati non-ChatML, quindi un ruolo "tool" verrebbe silenziosamente
    /// scartato su quei modelli. "user" con un prefisso esplicito funziona
    /// su ogni template.
    /// </summary>
    public static (string Role, string Content) BuildResultMessage(string toolName, string resultJson)
        => ("user", $"[tool_result name=\"{toolName}\"] {resultJson}");
}

/// <summary>
/// Output strutturato "best-effort": non essendoci un decoder a grammatica
/// (GBNF/JSON-schema constrained sampling) nel motore nativo, il JSON mode
/// qui e' realizzato con un'istruzione di prompt piu' un estrattore
/// tollerante che recupera il primo blocco JSON bilanciato e sintatticamente
/// valido anche se il modello ha aggiunto testo o code-fence attorno.
/// Non da' garanzie hard come un grammar sampler: se serve quel livello di
/// affidabilita', va implementato nel motore (vedi TODO in docs).
/// </summary>
public static class StructuredOutput
{
    public static string BuildJsonInstruction(string? jsonSchema = null)
    {
        var instr = "Respond with ONLY valid JSON and no other text: no markdown code fences, no explanation before or after.";
        if (!string.IsNullOrWhiteSpace(jsonSchema))
        {
            instr += " The JSON must conform to this schema:\n" + jsonSchema;
        }
        return instr;
    }

    /// <summary>
    /// Cerca nel testo il primo sottoblocco JSON (oggetto o array) bilanciato
    /// e sintatticamente valido, ignorando eventuale testo/markdown attorno.
    /// Restituisce null se nessun blocco parsa correttamente.
    /// </summary>
    public static string? TryExtractJson(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] != '{' && text[i] != '[') continue;
            var open = text[i];
            var close = open == '{' ? '}' : ']';
            var depth = 0;
            var inString = false;
            var escape = false;

            for (int j = i; j < text.Length; j++)
            {
                var c = text[j];
                if (inString)
                {
                    if (escape) escape = false;
                    else if (c == '\\') escape = true;
                    else if (c == '"') inString = false;
                    continue;
                }
                if (c == '"') { inString = true; continue; }
                if (c == open) depth++;
                else if (c == close)
                {
                    depth--;
                    if (depth == 0)
                    {
                        var candidate = text[i..(j + 1)];
                        try
                        {
                            using var _ = JsonDocument.Parse(candidate);
                            return candidate;
                        }
                        catch (JsonException)
                        {
                            break; // questo punto di partenza non e' JSON valido, prova il prossimo '{'/'['
                        }
                    }
                }
            }
        }
        return null;
    }
}
