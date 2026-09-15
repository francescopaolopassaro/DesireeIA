"""DesireeIA enums — 1:1 mapping of C ABI enums from abi.h."""

from enum import IntEnum


class Backend(IntEnum):
    CPU = 1
    CUDA = 2
    METAL = 3
    VULKAN = 4
    INTEL = 5
    AXELERA = 6


class ModelFormat(IntEnum):
    UNKNOWN = 0
    GGUF = 1
    SAFETENSORS = 2


class Quantization(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 4
    Q4K = 5
    Q5K = 6
    Q6K = 7


class Error(IntEnum):
    OK = 0
    INVALID_ARG = -1
    NOT_SUPPORTED = -2
    IO = -3
    PARSE = -4
    NO_MEM = -5
    UNDEFINED = -6


class SsdTierMode(IntEnum):
    OFF = 0
    AUTO = 1
    ALWAYS = 2


class SpecialToken(IntEnum):
    BOS = 0
    EOS = 1
    UNK = 2
    PAD = 3


class InferenceBackend(IntEnum):
    """Public backend enum. Unconfigured=0 lets the native plan auto-detect."""
    UNCONFIGURED = 0
    CPU = 1
    CUDA = 2
    METAL = 3
    VULKAN = 4
    INTEL = 5
    AXELERA = 6


class StreamType(IntEnum):
    IMAGE = 0
    VIDEO = 1
    FILE = 2
    DOCUMENT = 3
    AUDIO = 4
