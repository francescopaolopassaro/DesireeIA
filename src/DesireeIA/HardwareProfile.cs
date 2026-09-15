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
    bool Vulkan);