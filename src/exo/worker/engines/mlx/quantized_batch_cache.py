"""Batched quantized KV cache for mlx-lm's batch generation engine.

The pinned mlx-lm fork implements batching only for float KV caches
(``BatchKVCache``): ``QuantizedKVCache`` has no ``merge`` classmethod, so a
quantized per-request cache reaching ``_merge_caches`` raises. This module
provides ``BatchQuantizedKVCache`` — the full batch-cache API operating on
mlx's packed quantization layout, where ``keys``/``values`` are 3-tuples of
``(packed_data, scales, biases)`` sharing the sequence axis — and installs a
``merge`` classmethod onto ``QuantizedKVCache`` so quantized per-request
caches can enter the batch engine.

Attention needs no changes: ``mlx_lm.models.base.scaled_dot_product_attention``
routes any cache exposing ``.bits``/``.group_size`` through the quantized SDPA,
which is already batch-shape agnostic.
"""

from typing import TYPE_CHECKING, Callable, Protocol, cast, final

import mlx.core as mx
from mlx_lm.models.base import (
    create_causal_mask,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.models.cache import QuantizedKVCache

if TYPE_CHECKING:
    _CacheBase = object
else:
    from mlx_lm.models.cache import _BaseCache as _CacheBase

QuantizedTensor = tuple[mx.array, mx.array, mx.array]


class QuantizedCacheView(Protocol):
    """The runtime attributes of ``QuantizedKVCache`` missing from its stubs."""

    keys: QuantizedTensor | None
    values: QuantizedTensor | None
    offset: int
    group_size: int
    bits: int


def quantized_cache_view(cache: QuantizedKVCache) -> QuantizedCacheView:
    return cast(QuantizedCacheView, cast(object, cache))


def _map_members(
    tensor: QuantizedTensor, function: Callable[[mx.array], mx.array]
) -> QuantizedTensor:
    return (function(tensor[0]), function(tensor[1]), function(tensor[2]))


def _dynamic_roll(x: mx.array, shifts: mx.array, axis: int) -> mx.array:
    # Same as mlx_lm.models.cache.dynamic_roll (absent from the stubs).
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    return mx.take_along_axis(x, idx, axis=axis)


def _causal_mask(
    n: int, offset: int, left_padding: mx.array, window_size: int | None
) -> mx.array:
    return cast(
        mx.array,
        create_causal_mask(
            n, offset=offset, left_padding=left_padding, window_size=window_size
        ),
    )


@final
class BatchQuantizedKVCache(_CacheBase):
    step = 256

    def __init__(
        self, left_padding: list[int], group_size: int = 64, bits: int = 8
    ) -> None:
        self.keys: QuantizedTensor | None = None
        self.values: QuantizedTensor | None = None
        self.left_padding: mx.array = mx.array(left_padding)
        self.offset: mx.array = mx.array([-padding for padding in left_padding])
        self.group_size = group_size
        self.bits = bits
        self._idx = 0
        self._right_padding: mx.array | None = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[QuantizedTensor, QuantizedTensor]:
        batch_size, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        previous_index = self._idx

        if (
            self.keys is None
            or self.values is None
            or (previous_index + num_steps) > self.keys[0].shape[-2]
        ):
            elements_per_int = 8 * mx.uint32.size // self.bits
            growth = (self.step + num_steps - 1) // self.step * self.step
            base_shape = (batch_size, n_kv_heads, growth)

            def empty_quantized(dimension: int) -> QuantizedTensor:
                return (
                    mx.zeros(
                        (*base_shape, dimension // elements_per_int), dtype=mx.uint32
                    ),
                    mx.zeros(
                        (*base_shape, dimension // self.group_size), dtype=keys.dtype
                    ),
                    mx.zeros(
                        (*base_shape, dimension // self.group_size), dtype=keys.dtype
                    ),
                )

            if self.keys is not None and self.values is not None:
                existing_keys, existing_values = self.keys, self.values
                if previous_index % self.step != 0:

                    def clip(member: mx.array) -> mx.array:
                        return member[..., :previous_index, :]

                    existing_keys = _map_members(existing_keys, clip)
                    existing_values = _map_members(existing_values, clip)

                def expand(member: mx.array) -> mx.array:
                    padding = mx.zeros(
                        (*base_shape, member.shape[-1]), dtype=member.dtype
                    )
                    return mx.concatenate([member, padding], axis=-2)

                self.keys = _map_members(existing_keys, expand)
                self.values = _map_members(existing_values, expand)
            else:
                self.keys = empty_quantized(k_head_dim)
                self.values = empty_quantized(v_head_dim)

        self.offset = self.offset + num_steps
        self._idx += num_steps

        quantized_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        quantized_values = mx.quantize(
            values, group_size=self.group_size, bits=self.bits
        )
        for member_index in range(3):
            self.keys[member_index][..., previous_index : self._idx, :] = (
                quantized_keys[member_index]
            )
            self.values[member_index][..., previous_index : self._idx, :] = (
                quantized_values[member_index]
            )

        # Ensure offset and left_padding are evaluated with the cache contents.
        self.keys = (
            mx.depends(self.keys[0], [self.offset, self.left_padding]),
            self.keys[1],
            self.keys[2],
        )

        def visible(member: mx.array) -> mx.array:
            return member[..., : self._idx, :]

        return _map_members(self.keys, visible), _map_members(self.values, visible)

    def prepare(
        self,
        *,
        left_padding: list[int] | None = None,
        lengths: list[int] | None = None,
        right_padding: list[int] | None = None,
    ) -> None:
        del lengths  # accepted for BatchKVCache API parity
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            padding = mx.array(left_padding)
            self.left_padding = self.left_padding + padding
            self.offset = self.offset - padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self) -> None:
        if (
            self._right_padding is not None
            and self.keys is not None
            and self.values is not None
        ):
            padding = self._right_padding

            def roll(member: mx.array) -> mx.array:
                return _dynamic_roll(member, padding[:, None], axis=2)

            self.keys = _map_members(self.keys, roll)
            self.values = _map_members(self.values, roll)
            self.offset = self.offset - padding
            self.left_padding = self.left_padding + padding
            self._right_padding = None

    @property
    def state(
        self,
    ) -> tuple[QuantizedTensor, QuantizedTensor, mx.array, mx.array]:
        assert self.keys is not None and self.values is not None
        keys, values = self.keys, self.values
        if self._idx < keys[0].shape[2]:

            def visible(member: mx.array) -> mx.array:
                return member[..., : self._idx, :]

            keys = _map_members(keys, visible)
            values = _map_members(values, visible)
        return keys, values, self.offset, self.left_padding

    @state.setter
    def state(
        self, v: tuple[QuantizedTensor, QuantizedTensor, mx.array, mx.array]
    ) -> None:
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys[0].shape[2]

    @property
    def meta_state(self) -> tuple[str, ...]:
        return tuple(map(str, (self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v: tuple[str, ...]) -> None:
        self.group_size, self.bits = map(int, v)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    def make_mask(
        self, n: int, return_array: bool = False, window_size: int | None = None
    ) -> mx.array:
        del return_array  # accepted for BatchKVCache API parity
        return _causal_mask(
            n,
            offset=self._idx,
            left_padding=self.left_padding,
            window_size=window_size,
        )

    def filter(self, batch_indices: list[int] | mx.array) -> None:
        """In-place filter to keep just the given batch rows."""
        if self.keys is not None and self.values is not None:

            def take(member: mx.array) -> mx.array:
                return member[batch_indices]

            self.keys = _map_members(self.keys, take)
            self.values = _map_members(self.values, take)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_padding = int(self.left_padding.min().item())
        if min_left_padding > 0:
            if self.keys is not None and self.values is not None:

                def shift(member: mx.array) -> mx.array:
                    return member[..., min_left_padding:, :]

                self.keys = _map_members(self.keys, shift)
                self.values = _map_members(self.values, shift)
            self._idx -= min_left_padding
            self.left_padding = self.left_padding - min_left_padding

    def extend(self, other: "BatchQuantizedKVCache") -> None:
        """In-place extend this cache with the other cache."""
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        template_keys = self.keys if self.keys is not None else other.keys
        template_values = self.values if self.values is not None else other.values
        assert template_keys is not None and template_values is not None
        length_self = self.keys[0].shape[2] if self.keys is not None else 0
        length_other = other.keys[0].shape[2] if other.keys is not None else 0
        max_size = max(length_self, length_other)

        def pad_cache(
            cache: "BatchQuantizedKVCache",
        ) -> tuple[QuantizedTensor, QuantizedTensor, mx.array, mx.array]:
            assert template_keys is not None and template_values is not None
            keys, values = cache.keys, cache.values
            if keys is None or values is None:
                row_count = cache.offset.shape[0]

                def empty_like_member(member: mx.array) -> mx.array:
                    return mx.zeros(
                        (row_count, member.shape[1], 0, member.shape[3]),
                        dtype=member.dtype,
                    )

                keys = _map_members(template_keys, empty_like_member)
                values = _map_members(template_values, empty_like_member)
            left = max_idx - cache._idx
            right = max_size - keys[0].shape[2] - left
            if right < 0:

                def clip(member: mx.array) -> mx.array:
                    return member[..., :right, :]

                keys = _map_members(keys, clip)
                values = _map_members(values, clip)
                right = 0
            if left != 0 or right != 0:
                widths = [(0, 0), (0, 0), (left, right), (0, 0)]

                def pad(member: mx.array) -> mx.array:
                    return mx.pad(member, widths)

                keys = _map_members(keys, pad)
                values = _map_members(values, pad)
            return keys, values, cache.offset, cache.left_padding + left

        keys_a, values_a, offset_a, left_padding_a = pad_cache(self)
        keys_b, values_b, offset_b, left_padding_b = pad_cache(other)
        self.keys = (
            mx.concatenate([keys_a[0], keys_b[0]]),
            mx.concatenate([keys_a[1], keys_b[1]]),
            mx.concatenate([keys_a[2], keys_b[2]]),
        )
        self.values = (
            mx.concatenate([values_a[0], values_b[0]]),
            mx.concatenate([values_a[1], values_b[1]]),
            mx.concatenate([values_a[2], values_b[2]]),
        )
        self.offset = mx.concatenate([offset_a, offset_b])
        self.left_padding = mx.concatenate([left_padding_a, left_padding_b])
        self._idx = max_idx

    def extract(self, idx: int) -> QuantizedKVCache:
        assert self.keys is not None and self.values is not None
        cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)
        padding = int(self.left_padding[idx].item())

        def slice_row(member: mx.array) -> mx.array:
            return mx.contiguous(member[idx : idx + 1, :, padding : self._idx])

        view = quantized_cache_view(cache)
        view.keys = _map_members(self.keys, slice_row)
        view.values = _map_members(self.values, slice_row)
        view.offset = self._idx - padding
        return cache

    @classmethod
    def merge(cls, caches: list[QuantizedKVCache]) -> "BatchQuantizedKVCache":
        views = [quantized_cache_view(cache) for cache in caches]
        lengths = [view.offset for view in views]
        max_length = max(lengths)
        group_size = views[0].group_size
        bits = views[0].bits

        # No cache has content so make an empty one
        if max_length == 0:
            return cls([0] * len(views), group_size=group_size, bits=bits)

        padding = [max_length - length for length in lengths]
        batch_size = len(views)
        non_empty = [view for view in views if view.keys is not None]

        def members_of(
            view: QuantizedCacheView, of_values: bool
        ) -> QuantizedTensor:
            tensor = view.values if of_values else view.keys
            assert tensor is not None
            return tensor

        head_count = max(
            members_of(view, of_values=False)[0].shape[1] for view in non_empty
        )

        def merged_member(member_index: int, of_values: bool) -> mx.array:
            members = [
                members_of(view, of_values)[member_index] for view in non_empty
            ]
            dimension = max(member.shape[3] for member in members)
            return mx.zeros(
                (batch_size, head_count, max_length, dimension),
                dtype=members[0].dtype,
            )

        keys = (
            merged_member(0, of_values=False),
            merged_member(1, of_values=False),
            merged_member(2, of_values=False),
        )
        values = (
            merged_member(0, of_values=True),
            merged_member(1, of_values=True),
            merged_member(2, of_values=True),
        )
        for row, (pad_amount, view) in enumerate(zip(padding, views, strict=True)):
            if view.keys is None or view.values is None:
                continue
            for member_index in range(3):
                keys[member_index][
                    row : row + 1, :, pad_amount : pad_amount + view.offset
                ] = view.keys[member_index][..., : view.offset, :]
                values[member_index][
                    row : row + 1, :, pad_amount : pad_amount + view.offset
                ] = view.values[member_index][..., : view.offset, :]

        merged = cls(padding, group_size=group_size, bits=bits)
        merged.keys = keys
        merged.values = values
        merged.offset = merged.offset + max_length
        merged._idx = max_length
        return merged

    def size(self) -> int:
        return self._idx

    def empty(self) -> bool:
        return self.keys is None

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return sum(member.nbytes for member in (*self.keys, *self.values))


def install_batch_quantized_kv_cache() -> None:
    """Give ``QuantizedKVCache`` a ``merge`` classmethod so mlx-lm's batch
    engine (``_merge_caches``) can batch quantized per-request caches."""
    if getattr(QuantizedKVCache, "merge", None) is not None:
        return

    def merge(
        cls: type[QuantizedKVCache], caches: list[QuantizedKVCache]
    ) -> BatchQuantizedKVCache:
        del cls
        return BatchQuantizedKVCache.merge(caches)

    QuantizedKVCache.merge = classmethod(merge)  # pyright: ignore[reportAttributeAccessIssue]

    # Register in mlx_lm's cache module namespace so state round-trips that
    # resolve classes via globals() (e.g. CacheList.from_state) can find it.
    import mlx_lm.models.cache as mlx_cache_module

    if not hasattr(mlx_cache_module, "BatchQuantizedKVCache"):
        mlx_cache_module.BatchQuantizedKVCache = (  # pyright: ignore[reportAttributeAccessIssue]
            BatchQuantizedKVCache
        )
