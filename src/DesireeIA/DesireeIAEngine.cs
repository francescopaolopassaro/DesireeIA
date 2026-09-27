// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

namespace DesireeIA;

using System.Runtime.InteropServices;

public static class DesireeIAEngine
{
    public static HardwareProfile DetectHardware()
    {
        if (Native.NativeMethods.desireeia_probe_hw(out var hw) != Native.NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException("Hardware probe failed.");
        }

        return new HardwareProfile(
            CpuThreads: hw.CpuThreads,
            Avx: hw.CpuHasAvx != 0,
            Avx2: hw.CpuHasAvx2 != 0,
            Avx512: hw.CpuHasAvx512 != 0,
            Neon: hw.CpuHasNeon != 0,
            CudaDeviceCount: hw.CudaDeviceCount,
            RamTotalMb: hw.RamTotalMb,
            RamFreeMb: hw.RamFreeMb,
            IntelGpuCount: hw.IntelGpuCount,
            AxeleraDeviceCount: hw.AxeleraDeviceCount,
            Metal: hw.HasMetal != 0,
            Vulkan: hw.HasVulkan != 0);
    }

    public static ExecutionPlan BuildPlan(string? modelPath, ExecutionPlan? overrides = null)
    {
        var hw = DetectHardware();
        Native.NativeMethods.Plan nativePlan;

        var err = Native.NativeMethods.Error.Ok;
        IntPtr overridePtr = IntPtr.Zero;
        try
        {
            var nativeHw = NativeMethodsHw(hw);
            if (overrides is not null)
            {
                overridePtr = Marshal.AllocHGlobal(Marshal.SizeOf<Native.NativeMethods.Plan>());
                Marshal.StructureToPtr(overrides.ToNative(), overridePtr, false);
            }

            err = Native.NativeMethods.desireeia_make_plan(in nativeHw,
                                                       modelPath,
                                                       overridePtr,
                                                       out nativePlan);
        }
        finally
        {
            if (overridePtr != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(overridePtr);
            }
        }
        if (err != Native.NativeMethods.Error.Ok)
        {
            throw new InvalidOperationException($"Plan build failed: {err}");
        }

        return new ExecutionPlan
        {
            Backend = (InferenceBackend)nativePlan.Backend,
            Format = (ModelFormat)nativePlan.Format,
            DenseQuantization = (Quantization)nativePlan.DenseQuant,
            ExpertQuantization = (Quantization)nativePlan.ExpertQuant,
            ThreadCount = nativePlan.NThreads,
            RamBudgetMb = nativePlan.RamBudgetMb,
            ExpertCacheSize = nativePlan.ExpertCacheCount,
            ExpertPrefetch = nativePlan.ExpertPrefetchEnabled != 0,
            KvCompression = nativePlan.KvCompressionEnabled != 0,
            ExpertPinLearning = nativePlan.ExpertPinEnabled != 0,
            PrefetchDepth = nativePlan.ExpertPrefetchDepth,
            BatchUnion = nativePlan.BatchUnionEnabled != 0,
            DualSsd = nativePlan.DualSsdEnabled != 0
        };
    }

    /// <summary>
    /// Human-readable line with the cumulative time per matmul kernel type
    /// (activation quantization, Q4_0/Q4_K/Q6_K, float fallback).
    /// These are process-global counters: use <see cref="ProfileReset"/>
    /// before a measurement to isolate it to a window (e.g. just the decode
    /// loop of a benchmark).
    /// </summary>
    public static string ProfileDump()
    {
        var buf = new byte[2048];
        Native.NativeMethods.desireeia_profile_dump(buf, (nuint) buf.Length);
        var len = Array.IndexOf(buf, (byte) 0);
        return System.Text.Encoding.UTF8.GetString(buf, 0, len < 0 ? buf.Length : len);
    }

    public static void ProfileReset() => Native.NativeMethods.desireeia_profile_reset();

    /// <summary>Interface version of the loaded native engine (0 when it predates the query).</summary>
    public static int NativeAbiVersion
    {
        get
        {
            try { return Native.NativeMethods.desireeia_abi_version(); }
            catch (EntryPointNotFoundException) { return 0; }
        }
    }

    /// <summary>
    /// Throws a clear error when the native engine is older than this wrapper, instead of a
    /// missing-entry-point crash on the first new call.
    /// </summary>
    public static void EnsureAbi()
    {
        int v = NativeAbiVersion;
        if (v < Native.NativeMethods.AbiVersion)
            throw new InvalidOperationException(
                $"DesireeIA native engine interface {v} is older than this wrapper ({Native.NativeMethods.AbiVersion}): " +
                "update the native library that ships with the package.");
    }

    /// <summary>Text of the last native error raised on the calling thread.</summary>
    public static string LastError() => Native.NativeMethods.LastError(IntPtr.Zero);

    /// <summary>GPUs the engine can use (empty without CUDA).</summary>
    public static IReadOnlyList<GpuInfo> Gpus()
    {
        var list = new List<GpuInfo>();
        int n;
        try { n = Native.NativeMethods.desireeia_gpu_count(); }
        catch (EntryPointNotFoundException) { return list; }
        for (int i = 0; i < n; i++)
        {
            if (Native.NativeMethods.desireeia_probe_gpu(i, out var g) != Native.NativeMethods.Error.Ok) continue;
            int len = Array.IndexOf(g.Name, (byte)0);
            string name = System.Text.Encoding.UTF8.GetString(g.Name, 0, len < 0 ? g.Name.Length : len);
            list.Add(new GpuInfo(i, name, (long)g.TotalBytes, (long)g.FreeBytes, g.CcMajor, g.CcMinor, g.Multiprocessors));
        }
        return list;
    }

    private static Native.NativeMethods.HwInfo NativeMethodsHw(HardwareProfile hw) => new()
    {
        CpuThreads = hw.CpuThreads,
        CpuHasAvx = hw.Avx ? 1 : 0,
        CpuHasAvx2 = hw.Avx2 ? 1 : 0,
        CpuHasAvx512 = hw.Avx512 ? 1 : 0,
        CpuHasNeon = hw.Neon ? 1 : 0,
        CudaDeviceCount = hw.CudaDeviceCount,
        RamTotalMb = hw.RamTotalMb,
        RamFreeMb = hw.RamFreeMb,
        HasMetal = hw.Metal ? 1 : 0,
        HasVulkan = 0,
        IntelGpuCount = hw.IntelGpuCount,
        AxeleraDeviceCount = hw.AxeleraDeviceCount
    };
}