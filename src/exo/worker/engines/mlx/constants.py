# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

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
# process) once memory pressure on any node crosses these thresholds.
PREFILL_ABORT_METAL_ACTIVE_FRACTION: float = 0.9
PREFILL_ABORT_SYSTEM_USED_FRACTION: float = 0.92

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
