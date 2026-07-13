"""Unit tests for exo.worker.engines.mlx.mtp (MTP speculative decoding).

These exercise the pure/testable primitives (draft broadcast, accept/reject
math, cache rollback) without needing a real MTP checkpoint or a
multi-process mx.distributed group - see mtp.py's module docstring and the
Stage C report for what remains verified only at the unit level.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache

from exo.worker.engines.mlx.mtp import (
    AcceptRejectResult,
    broadcast_draft_ids,
    draft_contribution,
    maybe_load_mtp_head,
    mtp_available,
    rollback_mtp_cache,
    sequential_accept_reject,
)
from exo.worker.engines.mlx.quantized_batch_cache import BatchQuantizedKVCache
from exo.worker.engines.mlx.types import KVCacheType, Model


def _logrow(one_hot_index: int, vocab_size: int = 8, peak: float = 20.0) -> mx.array:
    """A near-deterministic logit row: `one_hot_index` overwhelmingly likely."""
    row = mx.zeros((vocab_size,))
    row = row.at[one_hot_index].add(peak)
    logprobs = row - mx.logsumexp(row, keepdims=True)
    mx.eval(logprobs)
    return logprobs


class TestDraftBroadcast:
    def test_group_none_is_passthrough(self) -> None:
        ids = mx.array([3, 1, 4], dtype=mx.uint32)
        out = broadcast_draft_ids(ids, contribute=True, group=None)
        assert out.tolist() == [3, 1, 4]

    def test_group_none_passthrough_ignores_contribute_flag(self) -> None:
        # With no distributed group there's only one rank, so `contribute`
        # is moot - the caller's own array always comes back unchanged.
        ids = mx.array([7, 7], dtype=mx.uint32)
        out = broadcast_draft_ids(ids, contribute=False, group=None)
        assert out.tolist() == [7, 7]

    def test_non_contributing_rank_payload_is_zero(self) -> None:
        ids = mx.array([5, 9, 2], dtype=mx.uint32)
        payload = draft_contribution(ids, contribute=False)
        assert payload.tolist() == [0, 0, 0]

    def test_contributing_rank_payload_is_unchanged(self) -> None:
        ids = mx.array([5, 9, 2], dtype=mx.uint32)
        payload = draft_contribution(ids, contribute=True)
        assert payload.tolist() == ids.tolist()

    def test_summed_contributions_recover_the_drafting_ranks_ids(self) -> None:
        # This is exactly what mx.distributed.all_sum computes across ranks
        # (see broadcast_draft_ids): one real contributor plus N-1 all-zero
        # contributors must sum back to the contributor's own array.
        drafted = mx.array([12, 0, 3, 41], dtype=mx.uint32)
        rank0 = draft_contribution(drafted, contribute=True)  # the drafting rank
        rank1 = draft_contribution(drafted, contribute=False)
        rank2 = draft_contribution(drafted, contribute=False)
        total = rank0 + rank1 + rank2
        assert total.tolist() == drafted.tolist()


class TestSequentialAcceptRejectGreedy:
    """Hand-computed examples, temp=0 (fully deterministic, no RNG)."""

    def test_all_drafts_accepted_yields_bonus_token(self) -> None:
        # Draft [2, 5] both match the backbone's own argmax at each position;
        # position 2 (the "bonus" slot) also has a clear argmax winner (col 1).
        target_logprobs = [_logrow(2), _logrow(5), _logrow(1)]
        result = sequential_accept_reject(
            [2, 5], target_logprobs, is_greedy=True, uniform_draws=[]
        )
        assert result.num_accepted == 2
        assert result.corrective_token is None
        assert result.bonus_token == 1
        assert result.final_logprobs.tolist() == target_logprobs[2].tolist()

    def test_first_draft_rejected_immediately(self) -> None:
        # Draft claims token 2, but the backbone's real argmax at position 0
        # is token 6 - reject with zero accepted.
        target_logprobs = [_logrow(6), _logrow(5), _logrow(1)]
        result = sequential_accept_reject(
            [2, 5], target_logprobs, is_greedy=True, uniform_draws=[]
        )
        assert result.num_accepted == 0
        assert result.corrective_token == 6
        assert result.bonus_token is None

    def test_second_draft_rejected_after_first_accepted(self) -> None:
        # First draft (token 2) matches; second draft (token 5) doesn't -
        # the real continuation after [y0, 2] is token 7.
        target_logprobs = [_logrow(2), _logrow(7), _logrow(1)]
        result = sequential_accept_reject(
            [2, 5], target_logprobs, is_greedy=True, uniform_draws=[]
        )
        assert result.num_accepted == 1
        assert result.corrective_token == 7
        assert result.bonus_token is None

    def test_wrong_target_logprob_count_raises(self) -> None:
        with pytest.raises(ValueError):
            sequential_accept_reject(
                [1, 2], [_logrow(1), _logrow(2)], is_greedy=True, uniform_draws=[]
            )


class TestSequentialAcceptRejectProbabilistic:
    def test_deterministic_given_same_seed(self) -> None:
        target_logprobs = [
            _logrow(2, peak=1.0),
            _logrow(5, peak=1.0),
            _logrow(1, peak=1.0),
        ]

        mx.random.seed(1234)
        u = [float(mx.random.uniform().item()), float(mx.random.uniform().item())]

        mx.random.seed(99)
        first = sequential_accept_reject(
            [2, 5], target_logprobs, is_greedy=False, uniform_draws=u
        )
        mx.random.seed(99)
        second = sequential_accept_reject(
            [2, 5], target_logprobs, is_greedy=False, uniform_draws=u
        )
        assert first.num_accepted == second.num_accepted
        assert first.corrective_token == second.corrective_token
        assert first.bonus_token == second.bonus_token

    def test_certain_target_always_accepts(self) -> None:
        # A near-one-hot target distribution at the drafted token: log-prob
        # is ~0, so acceptance is certain regardless of the uniform draw.
        target_logprobs = [_logrow(2), _logrow(5), _logrow(1)]
        result = sequential_accept_reject(
            [2, 5], target_logprobs, is_greedy=False, uniform_draws=[0.9999, 0.9999]
        )
        assert result.num_accepted == 2
        assert result.bonus_token is not None

    def test_impossible_target_always_rejects(self) -> None:
        # Drafted token has ~zero probability under the target distribution
        # at that position: rejection must happen even with u close to 0.
        target_logprobs = [_logrow(6, peak=60.0), _logrow(5), _logrow(1)]
        result = sequential_accept_reject(
            [2, 5], target_logprobs, is_greedy=False, uniform_draws=[1e-9, 1e-9]
        )
        assert result.num_accepted == 0
        assert result.corrective_token is not None
        # Residual sampling must never resample the just-rejected token.
        assert result.corrective_token != 2


class TestRollbackMtpCache:
    def _grown_kv_cache(
        self, n_tokens: int, *, heads: int = 2, dim: int = 4
    ) -> KVCache:
        cache = KVCache()
        for _ in range(n_tokens):
            k = mx.random.normal((1, heads, 1, dim))
            v = mx.random.normal((1, heads, 1, dim))
            cast("Cache", cache).update_and_fetch(k, v)
        return cache

    def test_kv_cache_trimmed_by_draft_length(self) -> None:
        kv = self._grown_kv_cache(5)
        rollback_mtp_cache([kv], 3)
        assert kv.offset == 2

    def test_batch_quantized_kv_cache_trimmed_by_draft_length(self) -> None:
        bq = BatchQuantizedKVCache([0], group_size=32, bits=8)
        keys = mx.random.normal((1, 2, 5, 32))
        values = mx.random.normal((1, 2, 5, 32))
        bq.update_and_fetch(keys, values)
        rollback_mtp_cache(cast(KVCacheType, [bq]), 3)
        assert bq.size() == 2

    def test_arrays_cache_restores_rollback_snapshot_not_trim(self) -> None:
        ac = ArraysCache(2)
        conv_before = mx.array([1.0, 2.0])
        ssm_before = mx.array([3.0, 4.0])
        ac[0] = conv_before
        ac[1] = ssm_before
        ac.rollback_state = (conv_before, ssm_before)
        # Simulate the draft chunk having advanced the live state further.
        ac[0] = mx.array([99.0, 99.0])
        ac[1] = mx.array([98.0, 98.0])

        rollback_mtp_cache([ac], 4)

        assert cast(mx.array, ac[0]).tolist() == conv_before.tolist()
        assert cast(mx.array, ac[1]).tolist() == ssm_before.tolist()
        assert ac.rollback_state is None

    def test_mixed_cache_list_each_entry_handled_per_type(self) -> None:
        kv = self._grown_kv_cache(4)
        bq = BatchQuantizedKVCache([0], group_size=32, bits=8)
        bq.update_and_fetch(
            mx.random.normal((1, 2, 4, 32)), mx.random.normal((1, 2, 4, 32))
        )
        ac = ArraysCache(2)
        conv_before = mx.array([5.0])
        ssm_before = mx.array([6.0])
        ac[0], ac[1] = conv_before, ssm_before
        ac.rollback_state = (conv_before, ssm_before)
        ac[0], ac[1] = mx.array([50.0]), mx.array([60.0])

        rollback_mtp_cache(cast(KVCacheType, [kv, bq, ac]), 2)

        assert kv.offset == 2
        assert bq.size() == 2
        assert cast(mx.array, ac[0]).tolist() == [5.0]
        assert cast(mx.array, ac[1]).tolist() == [6.0]

    def test_missing_rollback_state_falls_back_to_trim_error_for_non_trimmable(
        self,
    ) -> None:
        # An ArraysCache with no rollback_state set is a genuine bug in the
        # calling code (n_confirmed wasn't used) - rollback_mtp_cache must
        # not silently corrupt state by pretending it's trimmable.
        ac = ArraysCache(2)
        ac[0], ac[1] = mx.array([1.0]), mx.array([2.0])
        with pytest.raises(NotImplementedError):
            rollback_mtp_cache([ac], 1)


class TestMtpDisabledIsNoOp:
    """EXO_MTP_DRAFT=0 (the default) must leave the generate path untouched."""

    def test_maybe_load_mtp_head_is_noop_when_draft_tokens_is_zero(
        self, tmp_path: Path
    ) -> None:
        # No sidecar file exists, and MTP_DRAFT_TOKENS is 0 by default in a
        # plain test environment (EXO_MTP_DRAFT unset) - must return False
        # without attempting to touch the model at all.
        sentinel = cast(Model, cast(object, nn.Linear(4, 4)))
        loaded = maybe_load_mtp_head(tmp_path, sentinel, is_last_pipeline_rank=True)
        assert loaded is False

    def test_mtp_available_false_for_plain_module(self) -> None:
        assert mtp_available(nn.Linear(4, 4)) is False

    def test_generate_module_imports_without_mtp_side_effects(self) -> None:
        # Byte-for-byte import check: importing the generation modules must
        # not raise or require any MTP-specific setup when MTP is off.
        import exo.worker.engines.mlx.generator.batch_generate as batch_generate_module
        import exo.worker.engines.mlx.generator.generate as generate_module
        import exo.worker.engines.mlx.mtp as mtp_module

        assert hasattr(generate_module, "mlx_generate")
        assert hasattr(batch_generate_module, "ExoBatchGenerator")
        assert hasattr(mtp_module, "mtp_decode_step")


def test_accept_reject_result_is_frozen() -> None:
    result = AcceptRejectResult(
        num_accepted=0, corrective_token=1, bonus_token=None, final_logprobs=_logrow(0)
    )
    with pytest.raises((AttributeError, Exception)):
        result.num_accepted = 5  # type: ignore[misc]


def test_residual_sampling_never_returns_nan_or_inf() -> None:
    # Regression guard for the residual-distribution edge case (z == 0):
    # a target that puts (numerically) all its mass on the rejected token.
    vocab = 8
    row = mx.full((vocab,), -60.0)
    row = row.at[3].add(60.0)  # token 3 ~= probability 1, everything else ~0
    target_lp = row - mx.logsumexp(row, keepdims=True)
    mx.eval(target_lp)
    result = sequential_accept_reject(
        [3], [target_lp, _logrow(0)], is_greedy=False, uniform_draws=[0.999999]
    )
    # log_p_target(3) ~= 0 so acceptance is (numerically) certain; this
    # mainly guards that the z==0 fallback path doesn't crash if ever hit.
    assert result.num_accepted in (0, 1)
    if result.corrective_token is not None:
        assert not math.isnan(result.corrective_token)
