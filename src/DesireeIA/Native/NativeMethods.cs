// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

using System.Runtime.InteropServices;

namespace DesireeIA.Native;

internal static class NativeMethods
{
    private const string Lib = "DesireeIALocaleEngine";

    [StructLayout(LayoutKind.Sequential)]
    internal struct HwInfo
    {
        public int CpuThreads;
        public int CpuHasAvx;
        public int CpuHasAvx2;
        public int CpuHasAvx512;
        public int CpuHasNeon;
        public int CudaDeviceCount;
        public int CudaTotalMb;
        public ulong RamTotalMb;
        public ulong RamFreeMb;
        public int HasMetal;
        public int HasVulkan;
        public int IntelGpuCount;
        public int AxeleraDeviceCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct Plan
    {
        public int Backend;
        public int Format;
        public int DenseQuant;
        public int ExpertQuant;
        public int NThreads;
        public ulong RamBudgetMb;
        public int ExpertCacheCount;
        public int ExpertPrefetchEnabled;
        public int KvCompressionEnabled;
        public int ExpertPinEnabled;
        public int ExpertPrefetchDepth;
        public int BatchUnionEnabled;
        public int DualSsdEnabled;
        // Mirrors desireeia_plan.ssd_tier_mode / .ssd_tier_cache_mb. Field
        // order and types must stay identical to the C struct in abi.h:
        // this is a sequential-layout blit, not a marshalled conversion, so
        // a mismatch here corrupts memory rather than failing loudly.
        public int SsdTierMode;
        public ulong SsdTierCacheMb;
    }

    internal enum SsdTierModeNative : int
    {
        Off = 0,
        Auto = 1,
        Always = 2
    }

    internal enum Backend : int
    {
        Cpu = 1,
        Cuda = 2,
        Metal = 3,
        Vulkan = 4,
        Intel = 5,
        Axelera = 6
    }

    internal enum Format : int
    {
        Unknown = 0,
        Gguf = 1,
        Safetensors = 2
    }

    internal enum Quant : int
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q8_0 = 4,
        Q4K = 5,
        Q5K = 6,
        Q6K = 7
    }

    internal enum Error : int
    {
        Ok = 0,
        InvalidArg = -1,
        NotSupported = -2,
        Io = -3,
        Parse = -4,
        NoMem = -5,
        Undefined = -6
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void LogCallback(int level, IntPtr msg, IntPtr user);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr desireeia_version();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void desireeia_set_logger(LogCallback cb, IntPtr user);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_probe_hw(out HwInfo info);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_make_plan(in HwInfo hw,
                                                  [MarshalAs(UnmanagedType.LPUTF8Str)] string? modelPath,
                                                  IntPtr overridePlan,
                                                  out Plan plan);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_create([MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
                                               in Plan plan,
                                               LogCallback? cb,
                                               IntPtr user,
                                               out IntPtr ctx);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_destroy(IntPtr ctx);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_predict(IntPtr ctx, int[] tokens, nuint nTokens, out int outToken);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_next_token(IntPtr ctx, out int outToken);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern nuint desireeia_context_size(IntPtr ctx);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_tokenize(IntPtr ctx,
                                                 [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
                                                 int addBos,
                                                 int[]? outIds,
                                                 nuint maxIds,
                                                 out nuint outCount);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_token_piece(IntPtr ctx, int id, byte[] outBuf, nuint bufSize);

    // BERT (encoder-only): per-token embedding, same query-size convention as
    // desireeia_tokenize (outEmbd=null to query outLen/outEmbdDim).
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_embed(IntPtr ctx,
                                              int[] tokens,
                                              nuint nTokens,
                                              float[]? outEmbd,
                                              nuint outCapacity,
                                              out nuint outLen,
                                              out uint outEmbdDim);

    // which: 0=BOS 1=EOS 2=UNK 3=PAD (desireeia_special_token in abi.h).
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_special_token_id(IntPtr ctx, int which, out int outId);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_is_eog_token(IntPtr ctx, int id, out int outIsEog);

    /// <summary>Corresponds to desireeia_sampling in abi.h (sequential layout).</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct Sampling
    {
        public float Temperature;
        public int TopK;
        public float TopP;
        public float PenaltyRepeat;
        public float PenaltyFreq;
        public float PenaltyPresent;
        public int PenaltyLastN;
        public uint Seed;
    }

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_set_sampling(IntPtr ctx, in Sampling parameters);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_get_sampling(IntPtr ctx, out Sampling outParams);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_load_lora_adapter(IntPtr ctx,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string loraGgufPath, float scale);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_clear_lora_adapters(IntPtr ctx);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_load_prerouter(IntPtr ctx,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_clear_prerouter(IntPtr ctx);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_set_prerouter_heuristic(IntPtr ctx, int enabled);

    // string[] with ArraySubType=LPUTF8Str isn't supported marshaling on
    // this runtime ("Invalid managed/unmanaged type combination"): string
    // arrays are passed as IntPtr[] to manually allocated UTF-8 pointers
    // (see LocalModel.ApplyChatTemplate, which builds and frees them).
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_apply_chat_template(
        IntPtr ctx,
        IntPtr[] roles,
        IntPtr[] contents,
        nuint nMessages,
        int addAssistant,
        byte[]? outBuf,
        nuint bufSize,
        out nuint outLen);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void desireeia_profile_dump(byte[] outBuf, nuint bufSize);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void desireeia_profile_reset();

    // ============================================================
    // Vision Module - P/Invoke Declarations
    // ============================================================

    [StructLayout(LayoutKind.Sequential)]
    internal struct VisionImage
    {
        public uint Width;
        public uint Height;
        public uint Channels;
        public IntPtr Data;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct VisionConfig
    {
        public int EmbeddingDim;
        public int PatchSize;
        public int ImageSize;
        public int NumHeads;
        public int NumLayers;
        public int ProjectionDim;
        public int HasEncoder;
    }

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_load_image(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        int expectedChannels,
        out VisionImage outImage);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void desireeia_free_image(ref VisionImage image);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr desireeia_vision_create(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        LogCallback? cb,
        IntPtr user);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void desireeia_vision_destroy(IntPtr ctx);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_vision_get_config(IntPtr ctx, out VisionConfig outConfig);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_vision_encode(
        IntPtr ctx,
        in VisionImage image,
        float[]? outEmbd,
        nuint outCapacity,
        out nuint outLen,
        out uint outDim);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_vision_preprocess(
        in VisionImage input,
        int targetSize,
        float[]? outPixels,
        nuint outCapacity,
        out nuint outLen);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_has_vision(IntPtr ctx, out int outHas);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_vision_token_count(IntPtr ctx, out int outCount);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_vision_image_token(IntPtr ctx, out int outId);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_vision_encode_ctx(
        IntPtr ctx,
        in VisionImage image,
        float[]? outEmbd,
        nuint outCapacity,
        out nuint outLen,
        out uint outDim);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Error desireeia_predict_image(
        IntPtr ctx,
        int[] tokens,
        nuint nTokens,
        float[] embd,
        nuint nEmbd,
        int imageToken,
        out int outToken);
}
