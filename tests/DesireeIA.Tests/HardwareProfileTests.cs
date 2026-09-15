// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

public class HardwareProfileTests
{
    private static bool NativeAvailable()
    {
        try
        {
            _ = DesireeIAEngine.DetectHardware();
            return true;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    [Fact]
    public void DetectHardware_ReturnsValidProfile()
    {
        if (!NativeAvailable()) return;

        var hw = DesireeIAEngine.DetectHardware();
        Assert.True(hw.CpuThreads > 0);
        Assert.True(hw.RamTotalMb > 0);
    }

    [Fact]
    public void BuildPlan_ReturnsSensibleDefaults()
    {
        if (!NativeAvailable()) return;

        var hw = DesireeIAEngine.DetectHardware();
        var plan = DesireeIAEngine.BuildPlan(null);

        InferenceBackend expected = hw.CudaDeviceCount > 0 ? InferenceBackend.Cuda
            : hw.IntelGpuCount > 0 ? InferenceBackend.Intel
            : hw.Metal ? InferenceBackend.Metal
            : hw.AxeleraDeviceCount > 0 ? InferenceBackend.Axelera
            : InferenceBackend.Cpu;

        Assert.True(plan.ThreadCount > 0);
        Assert.True(plan.RamBudgetMb > 0);
        Assert.Equal(expected, plan.Backend);
    }

    [Fact]
    public void BuildPlan_AppliesUserOverrides()
    {
        if (!NativeAvailable()) return;

        var overrides = new ExecutionPlan { Backend = InferenceBackend.Cuda };
        var plan = DesireeIAEngine.BuildPlan(null, overrides);
        Assert.Equal(InferenceBackend.Cuda, plan.Backend);
    }
}