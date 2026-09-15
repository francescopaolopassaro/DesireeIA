// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Diagnostics;
using DesireeIA;

try
{
    var hw = DesireeIAEngine.DetectHardware();
    Console.WriteLine($"cpus={hw.CpuThreads} ramTotal={hw.RamTotalMb}MB avx2={hw.Avx2} " +
                      $"cuda={hw.CudaDeviceCount} intelArc={hw.IntelGpuCount} axelera={hw.AxeleraDeviceCount} metal={hw.Metal}");

    var plan = DesireeIAEngine.BuildPlan(null);
    Console.WriteLine(plan);
}
catch (DllNotFoundException ex)
{
    Console.WriteLine($"native library not present: {ex.Message}");
    Console.WriteLine("build the native engine first (cmake + compiler required).");
}