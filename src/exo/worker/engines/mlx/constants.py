# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

import os


def _fraction_from_environment(variable_name: str, default: float) -> float:
    raw_value = os.environ.get(variable_name)
    if raw_value is None:
        return default
    return float(raw_value)


KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
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
PREFILL_ABORT_SYSTEM_USED_FRACTION: float = _fraction_from_environment(
    "EXO_PREFILL_ABORT_SYSTEM_USED_FRACTION", 0.92
)

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
