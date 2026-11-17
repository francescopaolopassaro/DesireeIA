// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

namespace DesireeIA;

public readonly record struct HardwareProfile(
    int CpuThreads,
    bool Avx,
    bool Avx2,
    bool Avx512,
    bool Neon,
    int CudaDeviceCount,
    ulong RamTotalMb,
    ulong RamFreeMb,
    int IntelGpuCount,
    int AxeleraDeviceCount,
    bool Metal,
    bool Vulkan,
    // The GPUs the engine can use, with total and FREE video memory at the moment of the probe. Null when not probed
    // (a profile built by hand): the planner then has no video-memory figure and keeps to the RAM budget.
    IReadOnlyList<GpuInfo>? Gpus = null);