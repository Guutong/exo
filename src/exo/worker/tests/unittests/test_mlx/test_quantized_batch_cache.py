from typing import Callable, cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import QuantizedKVCache

from exo.worker.engines.mlx.quantized_batch_cache import (
    BatchQuantizedKVCache,
    install_batch_quantized_kv_cache,
    quantized_cache_view,
)

GROUP_SIZE = 32
BITS = 8
HEADS = 2
HEAD_DIM = 64


def _filled_quantized_cache(token_count: int, seed: int) -> QuantizedKVCache:
    cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
    if token_count > 0:
        mx.random.seed(seed)
        keys = mx.random.normal((1, HEADS, token_count, HEAD_DIM))
        values = mx.random.normal((1, HEADS, token_count, HEAD_DIM))
        update_and_fetch = cast(
            "Callable[[mx.array, mx.array], object]", cache.update_and_fetch
        )
        _ = update_and_fetch(keys, values)
    return cache


def _dequantized_keys(cache: QuantizedKVCache) -> mx.array:
    view = quantized_cache_view(cache)
    assert view.keys is not None
    data, scales, biases = (member[..., : view.offset, :] for member in view.keys)
    return mx.dequantize(
        data, scales, biases, group_size=view.group_size, bits=view.bits
    )


def test_install_gives_quantized_cache_a_merge_classmethod() -> None:
    install_batch_quantized_kv_cache()
    merge = cast(
        "Callable[[list[QuantizedKVCache]], object] | None",
        getattr(QuantizedKVCache, "merge", None),
    )
    assert merge is not None
    merged = merge(
        [_filled_quantized_cache(8, seed=0), _filled_quantized_cache(4, seed=1)]
    )
    assert isinstance(merged, BatchQuantizedKVCache)


def test_merge_pads_and_preserves_lengths() -> None:
    short = _filled_quantized_cache(4, seed=1)
    long = _filled_quantized_cache(10, seed=2)
    merged = BatchQuantizedKVCache.merge([short, long])

    assert merged.size() == 10
    assert merged.bits == BITS and merged.group_size == GROUP_SIZE
    assert merged.offset.tolist() == [4, 10]
    assert merged.left_padding.tolist() == [6, 0]


def test_merge_then_extract_roundtrips_content() -> None:
    original = _filled_quantized_cache(10, seed=3)
    other = _filled_quantized_cache(4, seed=4)
    merged = BatchQuantizedKVCache.merge([original, other])

    extracted = merged.extract(0)
    assert isinstance(extracted, QuantizedKVCache)
    assert quantized_cache_view(extracted).offset == 10
    assert mx.allclose(
        _dequantized_keys(extracted), _dequantized_keys(original)
    ).item()


def test_batched_update_and_fetch_returns_quantized_tuples() -> None:
    merged = BatchQuantizedKVCache.merge(
        [_filled_quantized_cache(4, seed=5), _filled_quantized_cache(4, seed=6)]
    )
    mx.random.seed(7)
    new_keys = mx.random.normal((2, HEADS, 1, HEAD_DIM))
    new_values = mx.random.normal((2, HEADS, 1, HEAD_DIM))
    keys, values = merged.update_and_fetch(new_keys, new_values)

    assert len(keys) == 3 and len(values) == 3
    assert keys[0].shape[2] == 5
    assert merged.size() == 5
    assert merged.offset.tolist() == [5, 5]


def test_filter_keeps_selected_rows_and_strips_padding() -> None:
    merged = BatchQuantizedKVCache.merge(
        [_filled_quantized_cache(4, seed=8), _filled_quantized_cache(10, seed=9)]
    )
    merged.filter([0])

    assert merged.offset.tolist() == [4]
    assert merged.left_padding.tolist() == [0]
    assert merged.size() == 4


def test_extend_concatenates_batches() -> None:
    first = BatchQuantizedKVCache.merge([_filled_quantized_cache(6, seed=10)])
    second = BatchQuantizedKVCache.merge([_filled_quantized_cache(3, seed=11)])
    first.extend(second)

    assert first.offset.tolist() == [6, 3]
    assert first.size() == 6
    assert first.keys is not None and first.keys[0].shape[0] == 2


def test_trim_reduces_visible_length() -> None:
    merged = BatchQuantizedKVCache.merge([_filled_quantized_cache(8, seed=12)])
    trimmed = merged.trim(3)

    assert trimmed == 3
    assert merged.size() == 5
    assert merged.offset.tolist() == [5]


def test_make_mask_broadcasts_against_5d_quantized_scores() -> None:
    merged = BatchQuantizedKVCache.merge(
        [_filled_quantized_cache(4, seed=20), _filled_quantized_cache(10, seed=21)]
    )
    mask = merged.make_mask(1)
    # Quantized SDPA scores are (B, n_kv_heads, n_repeats, L_q, L_k).
    scores_shape = (2, HEADS, 8, 1, merged.size())
    broadcast = mx.broadcast_to(mask, scores_shape)
    assert broadcast.shape == scores_shape


def test_merge_of_empty_caches_makes_empty_batch() -> None:
    merged = BatchQuantizedKVCache.merge(
        [
            QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS),
            QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS),
        ]
    )
    assert merged.empty()
    assert merged.size() == 0
    assert merged.nbytes == 0


def test_extract_after_decode_matches_dequantized_content() -> None:
    merged = BatchQuantizedKVCache.merge(
        [_filled_quantized_cache(4, seed=13), _filled_quantized_cache(9, seed=14)]
    )
    mx.random.seed(15)
    for _ in range(2):
        merged.update_and_fetch(
            mx.random.normal((2, HEADS, 1, HEAD_DIM)),
            mx.random.normal((2, HEADS, 1, HEAD_DIM)),
        )

    extracted_short = merged.extract(0)
    extracted_long = merged.extract(1)
    assert quantized_cache_view(extracted_short).offset == 6
    assert quantized_cache_view(extracted_long).offset == 11


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
