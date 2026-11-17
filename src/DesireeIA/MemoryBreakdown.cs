// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

namespace DesireeIA;

/// <summary>
/// Where the memory of a loaded model goes (see <see cref="LocalModel.GetMemoryBreakdown"/>), in megabytes.
/// The weights and the KV cache are ESTIMATED from the model header and the reserved context.
/// </summary>
public sealed record MemoryBreakdown(
    string Device,
    long VideoTotalMb,
    long VideoFreeMb,
    long WeightsMb,
    long KvReservedMb,
    int ReservedPositions,
    long KvBytesPerToken)
{
    /// <summary>Video memory in use right now by everything on the device (this model, other processes, the desktop).</summary>
    public long VideoUsedMb => Math.Max(0, VideoTotalMb - VideoFreeMb);

    /// <summary>One line for a status bar or a log.</summary>
    public override string ToString() =>
        $"{(Device.Length > 0 ? Device + ": " : "")}video {VideoUsedMb}/{VideoTotalMb} MB used, weights ~{WeightsMb} MB, " +
        $"KV ~{KvReservedMb} MB for {ReservedPositions} positions ({KvBytesPerToken / 1024} KB/token)";
}
