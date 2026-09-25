"""DesireeIA Python wrapper — public API.

Usage::

    import desireeia

    # Everything picked for this machine and model (backend, threads, RAM,
    # sampling, context, reply length):
    model, config = desireeia.LocalModel.load_auto("model.gguf")

    # or by hand:
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
from .autoconfig import (
    AutoConfiguration,
    ModelTraits,
    auto_configure,
    compute_configuration,
    read_gguf_metadata,
)
from .vision import VisionImageWrapper
from .generation import (
    StopSequenceScanner,
    ToolCalling,
    StructuredOutput,
)

__version__ = "0.0.5"

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
    # auto-configuration
    "AutoConfiguration",
    "ModelTraits",
    "auto_configure",
    "compute_configuration",
    "read_gguf_metadata",
    # vision
    "VisionImageWrapper",
    # generation
    "StopSequenceScanner",
    "ToolCalling",
    "StructuredOutput",
]
