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