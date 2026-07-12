# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

import os


def _fraction_from_environment(variable_name: str, default: float) -> float:
    raw_value = os.environ.get(variable_name)
    if raw_value is None:
        return default
    return float(raw_value)


def _optional_bits_from_environment(
    variable_name: str, default: int | None
) -> int | None:
    raw_value = os.environ.get(variable_name)
    if raw_value is None:
        return default
    if raw_value.strip().lower() in ("", "none", "0"):
        return None
    return int(raw_value)


KV_GROUP_SIZE: int | None = 32
# Quantize the KV cache to this many bits (roughly 16/bits x context headroom).
# Off by default; enable per deployment with e.g. EXO_KV_BITS=8. Must be set
# identically on every node of the cluster.
KV_BITS: int | None = _optional_bits_from_environment("EXO_KV_BITS", None)
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None

DEFAULT_TOP_LOGPROBS: int = 5

# Abort prefill with a clean error (instead of letting Metal OOM abort the runner
# process) once memory pressure on any node crosses these thresholds. Overridable
# per deployment via environment variables of the same name prefixed with EXO_.
PREFILL_ABORT_METAL_ACTIVE_FRACTION: float = _fraction_from_environment(
    "EXO_PREFILL_ABORT_METAL_ACTIVE_FRACTION", 0.9
)
# 0.95: on 16GB nodes the exo Python processes (~3GB, torch import included)
# plus macOS leave a loaded model idling near 88% system-used, so 0.92 rejected
# prompts the cluster comfortably fits. Per-chunk cache eval + pool clearing
# removed the mid-chunk fp16 spikes that made higher thresholds crash before,
# and the Metal-active guard remains the hard backstop for wired memory.
PREFILL_ABORT_SYSTEM_USED_FRACTION: float = _fraction_from_environment(
    "EXO_PREFILL_ABORT_SYSTEM_USED_FRACTION", 0.95
)

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
