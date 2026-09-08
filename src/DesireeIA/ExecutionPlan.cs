namespace DesireeIA;

public sealed class ExecutionPlan
{
    public InferenceBackend Backend { get; init; } = InferenceBackend.Cpu;
    public ModelFormat Format { get; init; } = ModelFormat.Unknown;
    public Quantization DenseQuantization { get; init; } = Quantization.Q8_0;
    public Quantization ExpertQuantization { get; init; } = Quantization.Q4K;
    public int ThreadCount { get; init; } = Environment.ProcessorCount;
    public ulong RamBudgetMb { get; init; }
    public int ExpertCacheSize { get; init; } = 256;
    public bool ExpertPrefetch { get; init; } = true;
    public bool KvCompression { get; init; } = true;
    public bool ExpertPinLearning { get; init; } = true;
    public int PrefetchDepth { get; init; } = 1;
    public bool BatchUnion { get; init; } = true;
    public bool DualSsd { get; init; }

    /// <summary>
    /// How the SSD storage tier participates in serving model weights.
    /// <see cref="SsdTierMode.Auto"/> (the default) behaves exactly like
    /// <see cref="SsdTierMode.Off"/> for any model that fits the RAM budget,
    /// so the common case pays nothing for the tier being available.
    /// </summary>
    public SsdTierMode SsdTier { get; init; } = SsdTierMode.Auto;

    /// <summary>
    /// RAM the SSD tier may use for its own hot-block cache, in MiB.
    /// 0 derives it from <see cref="RamBudgetMb"/>. Ignored when
    /// <see cref="SsdTier"/> is <see cref="SsdTierMode.Off"/>.
    /// </summary>
    public ulong SsdTierCacheMb { get; init; }

    public override string ToString() =>
        $"backend={Backend} format={Format} dense={DenseQuantization} experts={ExpertQuantization} " +
        $"threads={ThreadCount} ram={RamBudgetMb}MB cache={ExpertCacheSize} pin={ExpertPinLearning} " +
        $"prefetchDepth={PrefetchDepth} batchUnion={BatchUnion} dualSsd={DualSsd} kvCompression={KvCompression} " +
        $"ssdTier={SsdTier}";

    internal Native.NativeMethods.Plan ToNative() => new()
    {
        Backend = (int)Backend,
        Format = (int)Format,
        DenseQuant = (int)DenseQuantization,
        ExpertQuant = (int)ExpertQuantization,
        NThreads = ThreadCount,
        RamBudgetMb = RamBudgetMb,
        ExpertCacheCount = ExpertCacheSize,
        ExpertPrefetchEnabled = ExpertPrefetch ? 1 : 0,
        KvCompressionEnabled = KvCompression ? 1 : 0,
        ExpertPinEnabled = ExpertPinLearning ? 1 : 0,
        ExpertPrefetchDepth = PrefetchDepth,
        BatchUnionEnabled = BatchUnion ? 1 : 0,
        DualSsdEnabled = DualSsd ? 1 : 0,
        SsdTierMode = (int)SsdTier,
        SsdTierCacheMb = SsdTierCacheMb
    };
}

/// <summary>
/// How the SSD storage tier participates in serving model weights.
/// </summary>
public enum SsdTierMode
{
    /// <summary>Never use the SSD tier; weights are served from RAM/CPU only.</summary>
    Off = 0,

    /// <summary>
    /// Use the SSD tier only when the model does not fit the RAM budget.
    /// For a model that fits, this is identical to <see cref="Off"/>.
    /// </summary>
    Auto = 1,

    /// <summary>
    /// Always stream through the SSD tier, even when the model would fit in
    /// RAM. Slower by design; useful for measuring the tier, or for running a
    /// model far larger than RAM where predictable streaming beats swapping.
    /// </summary>
    Always = 2
}