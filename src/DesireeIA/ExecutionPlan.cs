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

    public override string ToString() =>
        $"backend={Backend} format={Format} dense={DenseQuantization} experts={ExpertQuantization} " +
        $"threads={ThreadCount} ram={RamBudgetMb}MB cache={ExpertCacheSize} pin={ExpertPinLearning} " +
        $"prefetchDepth={PrefetchDepth} batchUnion={BatchUnion} dualSsd={DualSsd} kvCompression={KvCompression}";

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
        DualSsdEnabled = DualSsd ? 1 : 0
    };
}