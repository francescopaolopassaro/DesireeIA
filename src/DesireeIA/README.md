# DesireeIA

C# wrapper for **DesireeIA**, a local LLM inference engine (GGUF models,
CPU/CUDA backends). This package is a thin, safe binding over the native
engine, with prebuilt binaries bundled for every platform it was built for
- no separate native install step.

Full documentation, model compatibility notes and the CLI/server side of
the project live in the main repository:
https://github.com/francescopaolopassaro/DesireeIA

## Install

```
dotnet add package DesireeIA
```

## Quick start

```csharp
using DesireeIA;

var plan = DesireeIAEngine.BuildPlan("model.gguf");
using var model = LocalModel.Load("model.gguf", plan);

await foreach (var piece in model.StreamAsync(model.Tokenize("Hello!")!, new GenerateOptions { MaxTokens = 256 }))
{
    Console.Write(piece);
}
```

## Conversation session

A chat usually re-sends its whole history every turn. `LocalModel` keeps the
KV cache between calls and prefills only the tokens that are new since the
previous prompt, so each turn costs the new messages, not the whole
conversation again. It's on by default and exact: the output is identical to
a full prefill.

```csharp
model.SessionReuse = SessionReuseMode.Exact;        // default
// SessionReuseMode.IncludeGenerated: also reuse generated tokens (faster, not bit-identical)
// SessionReuseMode.Off: always prefill everything (behavior before 0.0.2)
Console.WriteLine(model.LastReusedTokens);           // prompt tokens taken from the cache
model.ResetSession();                                // start over
```

## Bundled platforms

Check the `runtimes/` folder inside the package for the exact platforms
bundled - at minimum `win-x64` and `linux-x64`. If your platform/arch
isn't bundled yet, loading the native library will fail with a clear
error rather than silently doing nothing.

- **Windows**: the engine DLL and its CUDA runtime (`cudart64_13.dll`,
  for GPU acceleration) are bundled. It also needs the **Microsoft
  Visual C++ Redistributable (x64)** on the system - already present on
  most Windows machines, but not bundled in this package. If loading
  fails with a missing-DLL error, install it from
  https://aka.ms/vs/17/release/vc_redist.x64.exe and retry.
- **Linux**: the engine `.so` only depends on `libstdc++`, `libgcc_s`
  and `glibc` - present on essentially every distribution by default.
  GPU acceleration is not packaged for Linux yet (CPU-only) - a known
  gap, not a silent limitation.
- **macOS**: not built yet (`osx-x64`/`osx-arm64` are not bundled).

## License

See `LICENSE` - proprietary, no modification or unauthorized integration,
no AI training/ingestion without explicit written consent from the
author.
