// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

namespace DesireeIA;

public enum ModelFormat
{
    Unknown = 0,
    Gguf = 1,
    Safetensors = 2
}

public enum InferenceBackend
{
    // 0 = unspecified: the native plan auto-detects it from the hardware
    // (desireeia_make_plan only fills plan.backend when it is 0, see
    // core/plan.cpp/ctx.cpp). ExecutionPlan.Backend uses this as its default
    // on purpose: a partial override like `new ExecutionPlan { ThreadCount =
    // t }` must not silently force Cpu on a machine with a real GPU just
    // because the backend wasn't being set.
    Unconfigured = 0,
    Cpu = 1,
    Cuda = 2,
    Metal = 3,
    Vulkan = 4,
    Intel = 5,
    Axelera = 6
}

public enum Quantization
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 4,
    Q4K = 5,
    Q5K = 6,
    Q6K = 7
}

/// <summary>
/// How <see cref="LocalModel.Predict"/> reuses the K/V cache of the previous call
/// (see <see cref="LocalModel.SessionReuse"/>).
/// </summary>
public enum SessionReuseMode
{
    /// <summary>Always prefill the whole prompt (behavior before 0.0.2).</summary>
    Off = 0,
    /// <summary>Reuse only positions computed by a previous prefill: output identical to a full prefill.</summary>
    Exact = 1,
    /// <summary>Also reuse generated tokens: more reuse, not bit-identical (decode and prefill round differently).</summary>
    IncludeGenerated = 2
}
