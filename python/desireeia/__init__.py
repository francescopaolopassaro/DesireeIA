"""DesireeIA Python wrapper — public API.

Usage::

    import desireeia

    plan = desireeia.build_plan("model.gguf")
    with desireeia.LocalModel.load("model.gguf", plan) as model:
        tokens = model.tokenize("Hello, how are you?")
        for piece in model.chat_stream([("user", "Hello!")]):
            print(piece, end="", flush=True)
"""

from .enums import (
    Backend,
    Error,
    InferenceBackend,
    ModelFormat,
    Quantization,
    SsdTierMode,
    SpecialToken,
    StreamType,
)
from .types import (
    ExecutionPlan,
    GenerateOptions,
    HardwareProfile,
    SamplingOptions,
    ToolCall,
    ToolDefinition,
    VisionConfig,
)
from .engine import (
    build_plan,
    detect_hardware,
    profile_dump,
    profile_reset,
    version,
)
from .model import LocalModel
from .vision import VisionImageWrapper
from .generation import (
    StopSequenceScanner,
    ToolCalling,
    StructuredOutput,
)

__version__ = "0.0.4"

__all__ = [
    # enums
    "Backend",
    "Error",
    "InferenceBackend",
    "ModelFormat",
    "Quantization",
    "SsdTierMode",
    "SpecialToken",
    "StreamType",
    # types
    "ExecutionPlan",
    "GenerateOptions",
    "HardwareProfile",
    "SamplingOptions",
    "ToolCall",
    "ToolDefinition",
    "VisionConfig",
    # engine
    "build_plan",
    "detect_hardware",
    "profile_dump",
    "profile_reset",
    "version",
    # model
    "LocalModel",
    # vision
    "VisionImageWrapper",
    # generation
    "StopSequenceScanner",
    "ToolCalling",
    "StructuredOutput",
]
