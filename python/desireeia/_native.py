"""ctypes bindings to DesireeIALocaleEngine — the C ABI bridge.

Mirrors src/DesireeIA/Native/NativeMethods.cs exactly. Struct field order and
types MUST stay identical to the C structs in abi.h: this is a sequential
blit, not a marshalled conversion.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from ctypes import (
    POINTER,
    c_char_p,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_float,
    c_void_p,
    CFUNCTYPE,
    Structure,
)

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIB_NAMES = {
    "win32": "DesireeIALocaleEngine.dll",
    "darwin": "libDesireeIALocaleEngine.dylib",
    "linux": "libDesireeIALocaleEngine.so",
}

# Python sys.platform -> (fname suffix, arch candidates in priority order).
# The bundle ships the engine under desireeia/lib/<RID>/ following the same
# .NET RID convention used by Build/runtimes/<RID>/native/ (win-x64,
# linux-x64, osx-x64, osx-arm64, ...).
_RID_BY_BASE = {
    "win": "win-{arch}",
    "osx": "osx-{arch}",
    "linux": "linux-{arch}",
}

_ARCH_MAP = {
    "win":    {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"},
    "osx":    {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"},
    "linux":  {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64", "armv7l": "arm", "armv8l": "arm64"},
}

def _candidate_rids() -> list[str]:
    """The .NET RIDs this machine could match, most specific first."""
    import platform
    sysname = platform.system().lower()
    machine_raw = platform.machine().lower()

    base = None
    for key in ("win", "osx", "linux", "windows", "darwin"):
        if sysname == key:
            base = {"windows": "win", "darwin": "osx"}.get(key, key)
            break
    if base is None:
        return []

    arch_map = _ARCH_MAP.get(base, {})
    arch = arch_map.get(machine_raw, "")
    if not arch:
        return []
    return [f"{base}-{arch}"]


def _find_lib() -> ctypes.CDLL:
    env = os.environ.get("DESIREEIA_NATIVE_LIB")
    if env:
        if not os.path.isfile(env):
            raise OSError(f"DESIREEIA_NATIVE_LIB points to a missing file: {env}")
        return _load_with_deps(env)

    name = _LIB_NAMES.get(sys.platform)
    if name is None:
        raise OSError(f"Unsupported platform: {sys.platform}")

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs: list[str] = []

    # 1. Bundled per-platform library: desireeia/lib/<RID>/<fname>
    for rid in _candidate_rids():
        search_dirs.append(os.path.join(pkg_dir, "lib", rid))

    # 2. Same directory as this package (development layout)
    search_dirs.append(pkg_dir)

    # 3. Alongside the Python package root
    search_dirs.append(os.path.dirname(pkg_dir))

    # 4. Plain ctypes search (system paths, LD_LIBRARY_PATH, PATH, ...)
    try:
        return _load_with_deps(name)
    except OSError:
        pass

    for d in search_dirs:
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate):
            return _load_with_deps(candidate)

    raise OSError(
        f"Cannot find {name}. Ship it next to the desireeia package, bundle it "
        f"under desireeia/lib/<RID>/ (Build/python.ps1 does this automatically), "
        f"set DESIREEIA_NATIVE_LIB to its path, or add its directory to "
        f"PATH / LD_LIBRARY_PATH."
    )


def _load_with_deps(path: str) -> ctypes.CDLL:
    """Load the engine, making sibling runtime DLLs (MinGW: libstdc++-6.dll
    and friends) discoverable: on Windows the loader does NOT search the
    engine's own directory for its dependencies, so the bundle dir must be
    registered as a DLL search dir first."""
    if sys.platform == "win32":
        dll_dir = os.path.dirname(os.path.abspath(path))
        if dll_dir and os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
    return ctypes.CDLL(path)

_lib: ctypes.CDLL | None = None
_lib_lock = threading.Lock()

def get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        _lib = _find_lib()
        _setup_signatures(_lib)
        return _lib

# ---------------------------------------------------------------------------
# Log callback type
# ---------------------------------------------------------------------------

LOG_CB = CFUNCTYPE(None, c_int32, c_char_p, c_void_p)

# ---------------------------------------------------------------------------
# C structs (sequential layout, matching abi.h exactly)
# ---------------------------------------------------------------------------

class HwInfo(Structure):
    _fields_ = [
        ("cpu_threads",        c_int32),
        ("cpu_has_avx",        c_int32),
        ("cpu_has_avx2",       c_int32),
        ("cpu_has_avx512",     c_int32),
        ("cpu_has_neon",       c_int32),
        ("cuda_device_count",  c_int32),
        ("cuda_total_mb",      c_int32),
        ("ram_total_mb",       c_uint64),
        ("ram_free_mb",        c_uint64),
        ("has_metal",          c_int32),
        ("has_vulkan",         c_int32),
        ("intel_gpu_count",    c_int32),
        ("axelera_device_count", c_int32),
    ]

class Plan(Structure):
    _fields_ = [
        ("backend",                  c_int32),
        ("format",                   c_int32),
        ("dense_quant",              c_int32),
        ("expert_quant",             c_int32),
        ("n_threads",                c_int32),
        ("ram_budget_mb",            c_uint64),
        ("expert_cache_count",       c_int32),
        ("expert_prefetch_enabled",  c_int32),
        ("kv_compression_enabled",   c_int32),
        ("expert_pin_enabled",       c_int32),
        ("expert_prefetch_depth",    c_int32),
        ("batch_union_enabled",      c_int32),
        ("dual_ssd_enabled",         c_int32),
        ("ssd_tier_mode",            c_int32),
        ("ssd_tier_cache_mb",        c_uint64),
    ]

class Sampling(Structure):
    _fields_ = [
        ("temperature",      c_float),
        ("top_k",            c_int32),
        ("top_p",            c_float),
        ("penalty_repeat",   c_float),
        ("penalty_freq",     c_float),
        ("penalty_present",  c_float),
        ("penalty_last_n",   c_int32),
        ("seed",             c_uint32),
    ]

class VisionImage(Structure):
    _fields_ = [
        ("width",    c_uint32),
        ("height",   c_uint32),
        ("channels", c_uint32),
        ("data",     c_void_p),
    ]

class VisionConfig(Structure):
    _fields_ = [
        ("embedding_dim",  c_int32),
        ("patch_size",     c_int32),
        ("image_size",     c_int32),
        ("num_heads",      c_int32),
        ("num_layers",     c_int32),
        ("projection_dim", c_int32),
        ("has_encoder",    c_int32),
    ]

# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------

def _setup_signatures(lib: ctypes.CDLL) -> None:
    # version
    lib.desireeia_version.argtypes = []
    lib.desireeia_version.restype = c_char_p

    # logger
    lib.desireeia_set_logger.argtypes = [LOG_CB, c_void_p]
    lib.desireeia_set_logger.restype = c_int32

    # hardware probe
    lib.desireeia_probe_hw.argtypes = [POINTER(HwInfo)]
    lib.desireeia_probe_hw.restype = c_int32

    # plan
    lib.desireeia_make_plan.argtypes = [
        POINTER(HwInfo), c_char_p, POINTER(Plan), POINTER(Plan)
    ]
    lib.desireeia_make_plan.restype = c_int32

    # create / destroy
    lib.desireeia_create.argtypes = [
        c_char_p, POINTER(Plan), LOG_CB, c_void_p, POINTER(c_void_p)
    ]
    lib.desireeia_create.restype = c_int32

    lib.desireeia_destroy.argtypes = [c_void_p]
    lib.desireeia_destroy.restype = c_int32

    # predict / next_token / context_size
    lib.desireeia_predict.argtypes = [
        c_void_p, POINTER(c_int32), c_uint64, POINTER(c_int32)
    ]
    lib.desireeia_predict.restype = c_int32

    lib.desireeia_next_token.argtypes = [c_void_p, POINTER(c_int32)]
    lib.desireeia_next_token.restype = c_int32

    lib.desireeia_context_size.argtypes = [c_void_p]
    lib.desireeia_context_size.restype = c_uint64

    lib.desireeia_context_length_trained.argtypes = [c_void_p]
    lib.desireeia_context_length_trained.restype = c_uint32

    # tokenize
    lib.desireeia_tokenize.argtypes = [
        c_void_p, c_char_p, c_int32,
        POINTER(c_int32), c_uint64, POINTER(c_uint64)
    ]
    lib.desireeia_tokenize.restype = c_int32

    # token_piece
    lib.desireeia_token_piece.argtypes = [
        c_void_p, c_int32, POINTER(ctypes.c_char), c_uint64
    ]
    lib.desireeia_token_piece.restype = c_int32

    # special_token_id
    lib.desireeia_special_token_id.argtypes = [
        c_void_p, c_int32, POINTER(c_int32)
    ]
    lib.desireeia_special_token_id.restype = c_int32

    # is_eog_token
    lib.desireeia_is_eog_token.argtypes = [
        c_void_p, c_int32, POINTER(c_int32)
    ]
    lib.desireeia_is_eog_token.restype = c_int32

    # embed
    lib.desireeia_embed.argtypes = [
        c_void_p, POINTER(c_int32), c_uint64,
        POINTER(c_float), c_uint64, POINTER(c_uint64), POINTER(c_uint32)
    ]
    lib.desireeia_embed.restype = c_int32

    # sampling
    lib.desireeia_set_sampling.argtypes = [c_void_p, POINTER(Sampling)]
    lib.desireeia_set_sampling.restype = c_int32

    lib.desireeia_get_sampling.argtypes = [c_void_p, POINTER(Sampling)]
    lib.desireeia_get_sampling.restype = c_int32

    # lora
    lib.desireeia_load_lora_adapter.argtypes = [c_void_p, c_char_p, c_float]
    lib.desireeia_load_lora_adapter.restype = c_int32

    lib.desireeia_clear_lora_adapters.argtypes = [c_void_p]
    lib.desireeia_clear_lora_adapters.restype = c_int32

    # prerouter
    lib.desireeia_load_prerouter.argtypes = [c_void_p, c_char_p]
    lib.desireeia_load_prerouter.restype = c_int32

    lib.desireeia_clear_prerouter.argtypes = [c_void_p]
    lib.desireeia_clear_prerouter.restype = c_int32

    lib.desireeia_set_prerouter_heuristic.argtypes = [c_void_p, c_int32]
    lib.desireeia_set_prerouter_heuristic.restype = c_int32

    # chat template
    lib.desireeia_apply_chat_template.argtypes = [
        c_void_p,
        POINTER(c_char_p), POINTER(c_char_p), c_uint64, c_int32,
        POINTER(ctypes.c_char), c_uint64, POINTER(c_uint64)
    ]
    lib.desireeia_apply_chat_template.restype = c_int32

    # profiling
    lib.desireeia_profile_dump.argtypes = [
        POINTER(ctypes.c_char), c_uint64
    ]
    lib.desireeia_profile_dump.restype = None

    lib.desireeia_profile_reset.argtypes = []
    lib.desireeia_profile_reset.restype = None

    # vision: image load/free
    lib.desireeia_load_image.argtypes = [
        c_char_p, c_int32, POINTER(VisionImage)
    ]
    lib.desireeia_load_image.restype = c_int32

    lib.desireeia_free_image.argtypes = [POINTER(VisionImage)]
    lib.desireeia_free_image.restype = None

    # vision: standalone context
    lib.desireeia_vision_create.argtypes = [c_char_p, LOG_CB, c_void_p]
    lib.desireeia_vision_create.restype = c_void_p

    lib.desireeia_vision_destroy.argtypes = [c_void_p]
    lib.desireeia_vision_destroy.restype = None

    lib.desireeia_vision_get_config.argtypes = [
        c_void_p, POINTER(VisionConfig)
    ]
    lib.desireeia_vision_get_config.restype = c_int32

    lib.desireeia_vision_encode.argtypes = [
        c_void_p, POINTER(VisionImage),
        POINTER(c_float), c_uint64, POINTER(c_uint64), POINTER(c_uint32)
    ]
    lib.desireeia_vision_encode.restype = c_int32

    lib.desireeia_vision_preprocess.argtypes = [
        POINTER(VisionImage), c_int32,
        POINTER(c_float), c_uint64, POINTER(c_uint64)
    ]
    lib.desireeia_vision_preprocess.restype = c_int32

    # vision: ctx-based multimodal
    lib.desireeia_has_vision.argtypes = [c_void_p, POINTER(c_int32)]
    lib.desireeia_has_vision.restype = c_int32

    lib.desireeia_vision_token_count.argtypes = [c_void_p, POINTER(c_int32)]
    lib.desireeia_vision_token_count.restype = c_int32

    lib.desireeia_vision_image_token.argtypes = [c_void_p, POINTER(c_int32)]
    lib.desireeia_vision_image_token.restype = c_int32

    lib.desireeia_vision_encode_ctx.argtypes = [
        c_void_p, POINTER(VisionImage),
        POINTER(c_float), c_uint64, POINTER(c_uint64), POINTER(c_uint32)
    ]
    lib.desireeia_vision_encode_ctx.restype = c_int32

    lib.desireeia_predict_image.argtypes = [
        c_void_p, POINTER(c_int32), c_uint64,
        POINTER(c_float), c_uint64, c_int32, POINTER(c_int32)
    ]
    lib.desireeia_predict_image.restype = c_int32
