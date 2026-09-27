namespace DesireeIA;

/// <summary>One GPU the engine can use (see <see cref="DesireeIAEngine.Gpus"/>).</summary>
public sealed record GpuInfo(int Index, string Name, long TotalBytes, long FreeBytes,
                             int ComputeMajor, int ComputeMinor, int Multiprocessors);
