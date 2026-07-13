"""Shared types for MLX-related functionality."""

from collections.abc import Sequence
from typing import Any, Literal, overload

from mlx import core as mx
from mlx import nn as nn
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

# This list contains one cache entry per transformer layer. Note: exo's
# BatchQuantizedKVCache (quantized_batch_cache.py) also appears in real cache
# lists whenever EXO_KV_BITS is set and generation is batched, but is
# deliberately left out of this union - this codebase's existing exhaustive
# `match` statements over KVCacheType predate it and special-case
# QuantizedKVCache assuming an int offset, which BatchQuantizedKVCache (an
# mx.array offset for the whole padded batch) does not share. Code that needs
# to accept a BatchQuantizedKVCache-containing list (e.g.
# exo.worker.engines.mlx.mtp.rollback_mtp_cache, which only calls the
# is_trimmable()/trim()/rollback_state subset every member here and
# BatchQuantizedKVCache both duck-type) casts explicitly at the call site
# instead of widening this union.
KVCacheType = Sequence[
    KVCache
    | RotatingKVCache
    | QuantizedKVCache
    | ArraysCache
    | CacheList
    | DeepseekV4Cache
]


# Model is a wrapper function to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function
class Model(nn.Module):
    layers: list[nn.Module]

    @overload
    def __call__(
        self,
        x: mx.array,
        cache: KVCacheType | None,
        input_embeddings: mx.array | None = None,
        *,
        return_hidden: Literal[False] = False,
        n_confirmed: int = 0,
    ) -> mx.array: ...
    @overload
    def __call__(
        self,
        x: mx.array,
        cache: KVCacheType | None,
        input_embeddings: mx.array | None = None,
        *,
        return_hidden: Literal[True],
        n_confirmed: int = 0,
    ) -> tuple[mx.array, mx.array]: ...
    def __call__(
        self,
        x: mx.array,
        cache: KVCacheType | None,
        input_embeddings: mx.array | None = None,
        *,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array | tuple[mx.array, mx.array]: ...

    # MTP (Multi-Token Prediction) speculative decoding - only present when
    # the checkpoint declares an MTP head (mlx_lm PR #990's qwen3_5.TextModel
    # contract) and, for the sidecar-checkpoint case, after
    # exo.worker.engines.mlx.mtp.maybe_load_mtp_head has run. Callers must
    # check exo.worker.engines.mlx.mtp.mtp_loaded(model) first.
    def mtp_forward(
        self, hidden_states: mx.array, next_token_ids: mx.array, mtp_cache: list[Any]
    ) -> mx.array: ...
    def make_mtp_cache(self) -> list[Any]: ...
