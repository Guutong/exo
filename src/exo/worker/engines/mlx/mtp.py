"""MTP (Multi-Token Prediction) speculative decoding.

Wires exo's decode loop up to the native MTP head ported into the pinned
mlx-lm fork (``mlx_lm.models.qwen3_5.MTPModule`` / ``TextModel.mtp_forward``,
derived from ml-explore/mlx-lm PR #990) and the sidecar-checkpoint loader
(``mlx_lm.models.mtp.load_mtp_head``) for checkpoints that ship the MTP head
in a separate ``mtp.safetensors`` file (e.g.
Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed).

Everything in this module is inert when ``EXO_MTP_DRAFT`` (see
``exo.worker.engines.mlx.constants.MTP_DRAFT_TOKENS``) is ``0`` (the
default): ``maybe_load_mtp_head`` returns ``False`` immediately and no other
function here is called from the generation path.

Decode-step algorithm (see ``mtp_decode_step``), one round per call:

  1. Bootstrap forward over the single last-committed token -> sample ``y0``
     with the task's real sampler/logits-processors. Identical on every rank
     (the pipeline already broadcasts the final hidden state / logits to
     every rank via ``PipelineLastLayer``'s all-gather - see
     ``auto_parallel.py``).
  2. The drafting rank (last pipeline stage; the only rank with the MTP head
     and a materialized ``embed_tokens``) chains ``model.mtp_forward`` up to
     ``max_draft`` times, greedily taking the argmax token each step, reusing
     the SAME backbone hidden state (``hidden`` at ``y0``) each call and
     letting the MTP head's own small attention cache (``mtp_cache``)
     accumulate context - this matches the API sketch this module was built
     against. Non-drafting ranks produce a zero-filled placeholder array of
     the same shape.
  3. The draft token ids are broadcast to every rank (``broadcast_draft_ids``)
     so every rank builds the identical ``[y0, d1, ..., dk]`` sequence.
  4. One joint pipelined forward verifies all ``k`` drafts at once
     (``n_confirmed=1``), reusing the same multi-token forward mechanism the
     prefill path already relies on. Every rank computes the identical
     accept/reject decision (``sequential_accept_reject``) since sampling is
     seeded identically everywhere - no extra collective is needed for
     agreement (a debug-only cross-rank check is available via
     ``assert_accept_agreement``).
  5. All ``k`` drafts accepted: cache state is already correct (the model's
     own chunked SSM update is exact over the whole draft chunk), no
     rollback needed; a bonus token is sampled "for free" from the verify
     forward's last position.
     Any rejection (including a 0-token accept): the ``GatedDeltaNet``/
     ``ArraysCache`` rollback contract this fork implements only supports a
     SINGLE confirm/draft split point (see ``GatedDeltaNet.__call__``'s
     ``n_confirmed`` handling) - there is no way to recover the SSM state
     after an arbitrary *partial* prefix of the draft chunk without
     re-deriving it. So on any rejection this function rolls the whole draft
     chunk back to the post-``y0`` snapshot (``rollback_mtp_cache``) and
     redoes a small plain forward over exactly the accepted prefix plus the
     corrective token to re-populate cache state. This is a deliberate
     deviation from a naive "trim by the rejected count" - see the module
     docstring on ``rollback_mtp_cache`` for why that would silently corrupt
     the SSM recurrence.

Known v1 limitations (see the integration eligibility check in
``ExoBatchGenerator``, batch_generate.py):
  - No hidden-state chaining across decode rounds: every round starts with a
    fresh bootstrap forward instead of reusing the previous round's verify
    forward hidden state (as ``mtp_generate_step`` does via its
    ``cache_commit`` trick). Correct, but costs one extra small forward per
    round versus the theoretical minimum.
  - ``logits_processors`` (repetition/presence/frequency penalty) are not
    applied to the draft/verify positions within a round, only implicitly
    absent since MTP eligibility requires no such processors be configured
    for the task. A task with these penalties configured must fall back to
    the normal per-token decode path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache

from exo.worker.engines.mlx.cache import is_non_trimmable_cache_entry
from exo.worker.engines.mlx.constants import MTP_DRAFT_TOKENS, MTP_QUANTIZE_HEAD
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

MTP_SIDECAR_FILENAME = "mtp.safetensors"


def _text_model_of(model: nn.Module) -> nn.Module:
    """Return the ``TextModel`` instance (the one exposing ``.mtp``) for either
    a bare ``qwen3_5.TextModel`` or an outer multimodal wrapper that has a
    ``language_model`` attribute (mirrors ``mlx_lm.models.mtp._text_model_of``,
    duplicated here so this module doesn't depend on mlx_lm internals)."""
    language_model = getattr(model, "language_model", None)
    return language_model if isinstance(language_model, nn.Module) else model


def mtp_available(model: nn.Module) -> bool:
    """True if ``model`` has (or could grow) an MTP head, i.e. exposes the
    ``mtp_forward``/``make_mtp_cache`` contract from mlx_lm PR #990."""
    return hasattr(model, "mtp_forward") and hasattr(model, "make_mtp_cache")


def mtp_loaded(model: nn.Module) -> bool:
    """True if an MTP head is actually attached and populated on this rank."""
    return hasattr(_text_model_of(model), "mtp")


def _materialize_embed_tokens(model: nn.Module) -> int:
    """Force-evaluate ``embed_tokens`` weights and return their byte size.

    On a pipeline rank other than rank 0, ``embed_tokens`` is called every
    decode step (see ``Qwen3_5TextModel.__call__``) but its *output* is
    immediately discarded and overwritten by the value received from the
    previous rank (``PipelineFirstLayer.__call__``) - so as long as nothing
    forces evaluation of that dead branch, the lazily-loaded embedding
    weights never actually materialize into Metal memory on that rank. MTP's
    ``mtp_forward`` needs a real, callable ``embed_tokens`` on the drafting
    rank (normally the last pipeline stage), so this forces it once at load
    time instead of paying for it unpredictably mid-generation.
    """
    text_model = _text_model_of(model)
    embed_tokens = cast(nn.Module, text_model.model.embed_tokens)  # type: ignore[attr-defined]
    mx.eval(embed_tokens.parameters())
    return _module_nbytes(embed_tokens)


def _module_nbytes(module: nn.Module) -> int:
    flattened = cast(list[tuple[str, mx.array]], tree_flatten(module.parameters()))
    return sum(value.nbytes for _, value in flattened)


def maybe_load_mtp_head(
    model_path: Path,
    model: Model,
    *,
    is_last_pipeline_rank: bool,
    model_id: str | None = None,
) -> bool:
    """Attach and populate the MTP head from a sidecar file, if enabled.

    Only the last pipeline-parallel rank (the one that will actually draft)
    loads anything; every other rank pays no extra memory. No-op (returns
    ``False``) unless ``EXO_MTP_DRAFT`` > 0, ``model`` supports MTP, and a
    sidecar ``mtp.safetensors`` (or the checkpoint-declared equivalent) is
    present next to the model's shards.

    The sidecar is never part of ``model.safetensors.index.json``, so exo's
    own shard-aware downloader (which only fetches files listed there) never
    fetches it - a ``hf_hub_download`` is attempted here as a last resort
    when it's missing and ``model_id`` is known.
    """
    if MTP_DRAFT_TOKENS <= 0:
        return False
    if not is_last_pipeline_rank:
        return False
    if not mtp_available(model):
        logger.info(
            "EXO_MTP_DRAFT set but model has no MTP-capable TextModel "
            f"(model_type={getattr(model, 'model_type', '?')}) - skipping MTP"
        )
        return False

    sidecar_path = model_path / MTP_SIDECAR_FILENAME
    if not sidecar_path.exists() and model_id is not None:
        logger.info(
            f"EXO_MTP_DRAFT set, {MTP_SIDECAR_FILENAME} missing from "
            f"{model_path} - fetching it from {model_id}"
        )
        try:
            from huggingface_hub import (
                hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
            )

            _ = hf_hub_download(
                repo_id=model_id,
                filename=MTP_SIDECAR_FILENAME,
                local_dir=model_path,
            )
        except Exception:
            logger.opt(exception=True).info(
                f"No {MTP_SIDECAR_FILENAME} available for {model_id} - skipping MTP"
            )

    if not sidecar_path.exists():
        logger.info(
            f"EXO_MTP_DRAFT set but no {MTP_SIDECAR_FILENAME} found in "
            f"{model_path} - skipping MTP"
        )
        return False

    from mlx_lm.models.mtp import load_mtp_head

    try:
        load_mtp_head(model_path, model)
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to load MTP sidecar head - continuing without MTP"
        )
        return False

    text_model = _text_model_of(model)
    mtp_module = cast(nn.Module, text_model.mtp)
    head_bytes_before = _module_nbytes(mtp_module)

    if MTP_QUANTIZE_HEAD:
        # Quantize only the MTP head's plain (bf16-native) Linear layers -
        # the sidecar's numbered-expert MoE weights are already INT4
        # (QuantizedLinear instances, which nn.Linear-only predicates skip).
        def _is_plain_linear(_path: str, module: nn.Module) -> bool:
            return isinstance(module, nn.Linear)

        nn.quantize(
            mtp_module,
            group_size=64,
            bits=8,
            class_predicate=_is_plain_linear,
        )
        mx.eval(mtp_module.parameters())

    head_bytes_after = _module_nbytes(mtp_module)
    embed_bytes = _materialize_embed_tokens(model)

    logger.info(
        "MTP head loaded: "
        f"head={head_bytes_after / 2**20:.1f}MiB "
        f"(pre-quantize-head {head_bytes_before / 2**20:.1f}MiB) "
        f"embed_tokens={embed_bytes / 2**20:.1f}MiB "
        f"draft_tokens={MTP_DRAFT_TOKENS} quantize_head={MTP_QUANTIZE_HEAD}"
    )
    return True


def assert_accept_agreement(
    num_accepted: int, group: mx.distributed.Group | None, *, verbose: bool
) -> None:
    """Debug-only cross-rank check that every rank reached the same
    accept/reject decision this round.

    Not called on the hot path by default: the decision is provably
    deterministic given identical logits/seed on every rank (see module
    docstring), so this collective is redundant in the common case. Gate it
    behind verbose logging (``-vv``) since it costs one extra
    all-gather every MTP decode round.
    """
    if not verbose or group is None:
        return
    gathered = mx.distributed.all_gather(
        mx.array([num_accepted]),
        group=group,
        stream=mx.default_stream(mx.Device(mx.cpu)),
    )
    mx.eval(gathered)
    values: list[int] = gathered.tolist()  # type: ignore[assignment]
    if len(set(values)) > 1:
        logger.error(f"MTP accept-count disagreement across ranks: {values}")


def draft_contribution(draft_ids: mx.array, *, contribute: bool) -> mx.array:
    """The payload a single rank contributes to the ``all_sum`` broadcast in
    ``broadcast_draft_ids``: the real draft ids on the drafting rank, or an
    all-zero array of the identical shape/dtype on every other rank so the
    collective sum equals the drafting rank's array everywhere. Factored out
    so the "only one non-zero contributor" invariant is testable without a
    real multi-process ``mx.distributed`` group: summing one real
    contribution with N-1 zero contributions (from N-1 calls with
    ``contribute=False``) must equal the real contribution.
    """
    return draft_ids if contribute else mx.zeros_like(draft_ids)


def broadcast_draft_ids(
    draft_ids: mx.array,
    *,
    contribute: bool,
    group: mx.distributed.Group | None,
) -> mx.array:
    """Broadcast the drafting rank's ``draft_ids`` to every rank.

    Follows the same "all_sum with only one contributor" trick used by
    ``mx_any``/``mx_all_gather_tasks`` (utils_mlx.py) rather than a real
    broadcast primitive: every non-contributing rank must pass an
    all-zero array of the identical shape/dtype so the collective sum
    equals the contributor's array on every rank.
    """
    if group is None:
        return draft_ids
    payload = draft_contribution(draft_ids, contribute=contribute)
    summed = mx.distributed.all_sum(
        payload, group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(summed)
    return summed


@dataclass(frozen=True)
class AcceptRejectResult:
    """Outcome of verifying a chain of drafted tokens against the backbone.

    ``num_accepted`` is the count of leading drafted tokens (0..k) that
    survived verification. Exactly one of ``bonus_token``/``corrective_token``
    is set: ``bonus_token`` when every drafted token was accepted (a free
    extra sample past the last draft), ``corrective_token`` when
    verification stopped early (the resampled replacement for the first
    rejected draft). ``final_logprobs`` is the distribution the
    bonus/corrective token was actually sampled from (full-vocab log-softmax
    for the bonus case; the log-residual distribution - or the raw target
    log-softmax in the greedy/degenerate case - for the corrective case),
    for accurate user-facing logprob reporting on that token.
    """

    num_accepted: int
    corrective_token: int | None
    bonus_token: int | None
    final_logprobs: mx.array


def sequential_accept_reject(
    drafted_token_ids: Sequence[int],
    target_logprobs: Sequence[mx.array],
    *,
    is_greedy: bool,
    uniform_draws: Sequence[float] = (),
) -> AcceptRejectResult:
    """Verify a chain of ``k`` drafted tokens against target logprobs.

    Generalizes the k=1 verification in the mlx-lm fork's
    ``mtp_generate_step`` (derived from ml-explore/mlx-lm PR #990, itself
    following Leviathan et al. 2022 / Chen et al. 2023's speculative
    decoding rejection sampling) to a chain of ``k`` drafts, verified
    strictly left to right, stopping at the first rejection.

    Because the MTP head's draft is greedy/argmax (a Dirac-delta draft
    distribution, log-prob 0 at the drafted token and -inf elsewhere), the
    general accept-probability formula ``min(1, p_target(d)/p_draft(d))``
    reduces to simply ``p_target(d)``, and the residual-sampling formula
    ``max(p_target - p_draft, 0)/Z`` reduces to "p_target with the drafted
    token's mass zeroed, renormalized".

    Args:
        drafted_token_ids: The ``k`` greedily-drafted token ids, in order.
        target_logprobs: ``k + 1`` full-vocabulary log-probability arrays
            from a single joint backbone forward over
            ``[y0, d_1, ..., d_k]``: ``target_logprobs[i]`` (``i < k``) is
            the target distribution used to verify ``drafted_token_ids[i]``;
            ``target_logprobs[k]`` is the "bonus" distribution sampled from
            only when every draft is accepted.
        is_greedy: Whether the target sampler is temperature-0 (argmax).
            When true, verification and the corrective/bonus token are both
            plain argmax - no ``uniform_draws`` needed.
        uniform_draws: ``k`` pre-drawn ``Uniform(0, 1)`` samples, one per
            verify position, used only when ``is_greedy`` is false.

    Returns:
        An ``AcceptRejectResult``.
    """
    k = len(drafted_token_ids)
    if len(target_logprobs) != k + 1:
        raise ValueError(
            f"Expected {k + 1} target logprob rows for {k} drafted tokens, "
            f"got {len(target_logprobs)}"
        )
    if not is_greedy and len(uniform_draws) != k:
        raise ValueError(f"Expected {k} uniform draws, got {len(uniform_draws)}")

    for i in range(k):
        target_lp = target_logprobs[i]
        drafted = drafted_token_ids[i]
        if is_greedy:
            target_argmax = int(mx.argmax(target_lp).item())
            if target_argmax == drafted:
                continue
            return AcceptRejectResult(
                num_accepted=i,
                corrective_token=target_argmax,
                bonus_token=None,
                final_logprobs=target_lp,
            )

        log_p_target_drafted = float(target_lp[drafted].item())
        accepted = log_p_target_drafted >= 0.0 or uniform_draws[i] < math.exp(
            log_p_target_drafted
        )
        if accepted:
            continue

        p_target = mx.exp(target_lp)
        vocab_index = mx.arange(p_target.shape[0])
        residual = mx.where(vocab_index == drafted, mx.zeros_like(p_target), p_target)
        residual = mx.maximum(residual, 0.0)
        z = float(residual.sum().item())
        if z > 0.0:
            residual_lp = mx.log(residual)
            corrective = int(mx.random.categorical(residual_lp.reshape(1, -1)).item())
            final_lp = residual_lp
        else:
            # Degenerate: the target puts (numerically) all mass on the
            # rejected draft anyway. Fall back to its own argmax.
            corrective = int(mx.argmax(target_lp).item())
            final_lp = target_lp
        return AcceptRejectResult(
            num_accepted=i,
            corrective_token=corrective,
            bonus_token=None,
            final_logprobs=final_lp,
        )

    # Every drafted token was accepted: sample a free bonus token from the
    # verify forward's last position.
    bonus_lp = target_logprobs[k]
    if is_greedy:
        bonus = int(mx.argmax(bonus_lp).item())
    else:
        bonus = int(mx.random.categorical(bonus_lp.reshape(1, -1)).item())
    return AcceptRejectResult(
        num_accepted=k,
        corrective_token=None,
        bonus_token=bonus,
        final_logprobs=bonus_lp,
    )


def rollback_mtp_cache(cache: KVCacheType, num_draft_positions: int) -> None:
    """Undo an entire rejected (or partially-rejected) MTP draft chunk.

    Precondition: ``cache`` was produced by a forward pass with
    ``n_confirmed=1`` over exactly ``[y0, d_1, ..., d_{num_draft_positions}]``
    - i.e. every ``ArraysCache`` (GatedDeltaNet SSM/conv state) entry carries
    a ``.rollback_state`` snapshot taken right after ``y0`` (see
    ``qwen3_5.GatedDeltaNet.__call__``'s ``n_confirmed`` handling in the
    mlx-lm fork).

    This restores *every* cache entry to the state right after ``y0``,
    discarding the *entire* draft chunk - even on a partial accept (some
    leading drafts verified correctly) - rather than trimming only the
    rejected suffix. GatedDeltaNet's chunked SSM update only exposes a
    snapshot at the single ``n_confirmed`` split point; there is no cheap
    way to recover "state after y0 plus exactly the first m accepted
    drafts" without re-deriving it (the SSM layers of positions 1..m are
    entangled with attention-layer outputs from every earlier layer, so it
    can't be recomputed in isolation from the KV cache alone). Callers
    handling a partial accept are expected to redo a small plain forward
    over the accepted prefix + corrective token afterwards to re-populate
    cache state correctly (see ``mtp_decode_step``).
    """
    for entry in cache:
        if isinstance(entry, ArraysCache) and entry.rollback_state is not None:
            conv_snapshot, ssm_snapshot = entry.rollback_state
            entry[0] = conv_snapshot
            entry[1] = ssm_snapshot
            entry.rollback_state = None
        elif is_non_trimmable_cache_entry(entry):
            raise NotImplementedError(
                f"MTP rollback does not support cache entry type {type(entry).__name__} "
                "(no rollback_state and not directly trimmable). Qwen3.5/3.6's "
                "GatedDeltaNet is the only non-trimmable layer type validated "
                "against the MTP rollback contract."
            )
        else:
            entry.trim(num_draft_positions)


@dataclass
class MtpStepResult:
    """Tokens produced by one ``mtp_decode_step`` round, in *feed order*.

    Matches ``mlx_lm.generate.GenerationBatch``'s reporting convention (see
    ``GenerationBatch._step``): a token is only reported once it has actually
    been fed to the model and the cache reflects it. ``token_ids[0]`` is
    therefore always the caller's ``last_committed_token_id`` (fed by this
    call's bootstrap forward), and ``logprobs[0]`` is therefore the caller's
    ``last_committed_logprobs`` (the distribution it was sampled from by
    a *previous* round/step - this function does not compute it). Every
    later entry is both fed and reported within this same call.
    ``next_last_committed_token_id``/``next_last_committed_logprobs`` are the
    NOT-yet-fed pending token (and the distribution it was sampled from) to
    pass as inputs to the following round.
    """

    token_ids: list[int]
    logprobs: list[mx.array]
    from_draft: list[bool]
    next_last_committed_token_id: int
    next_last_committed_logprobs: mx.array
    num_drafted: int
    num_accepted: int


def _sample_position(
    logits_row: mx.array,
    *,
    filter_chain: list[Callable[[mx.array], mx.array]],
    is_greedy: bool,
    temp: float,
) -> tuple[int, mx.array]:
    """Sample one token from a single position's raw logits.

    Returns ``(token_id, logprobs)`` where ``logprobs`` is the *unfiltered*
    full-vocabulary log-softmax (used both for user-facing logprob reporting
    and as the accept/reject target distribution).
    """
    logprobs = logits_row - mx.logsumexp(logits_row, axis=-1, keepdims=True)
    if is_greedy:
        token = int(mx.argmax(logprobs).item())
        return token, logprobs
    masked = logprobs
    for f in filter_chain:
        masked = f(masked[None]).squeeze(0)
    scaled = masked / temp
    token = int(mx.random.categorical(scaled.reshape(1, -1)).item())
    return token, logprobs


def _build_filter_chain(
    top_p: float, top_k: int, min_p: float, min_tokens_to_keep: int
) -> list[Callable[[mx.array], mx.array]]:
    """Typed wrapper around ``mlx_lm.sample_utils.make_sampler_chain`` (whose
    stub return type carries an untyped XTC cell we never use here)."""
    from mlx_lm.sample_utils import make_sampler_chain

    filter_chain, _xtc_cell = cast(
        "tuple[list[Callable[[mx.array], mx.array]], object | None]",
        make_sampler_chain(top_p, top_k, min_p, min_tokens_to_keep),
    )
    return filter_chain


def mtp_decode_step(
    model: Model,
    cache: KVCacheType,
    mtp_cache: list[Any],
    *,
    last_committed_token_id: int,
    last_committed_logprobs: mx.array,
    temp: float,
    top_p: float,
    top_k: int,
    min_p: float,
    min_tokens_to_keep: int,
    max_draft: int,
    is_drafting_rank: bool,
    group: mx.distributed.Group | None,
) -> MtpStepResult:
    """Run one MTP speculative-decode round. See module docstring for the
    full algorithm. Batch-size-1 only - callers are responsible for the
    "fall back to the normal path when >1 task is active" admission check,
    and for not calling this when ``logits_processors`` are configured for
    the task (repetition/presence/frequency penalties need per-token history
    threading through the verify positions that this function does not do -
    see the module docstring's known-limitations note).
    """
    is_greedy = temp == 0
    filter_chain: list[Callable[[mx.array], mx.array]] = (
        _build_filter_chain(top_p, top_k, min_p, min_tokens_to_keep)
        if not is_greedy
        else []
    )

    def sample(logits_row: mx.array) -> tuple[int, mx.array]:
        return _sample_position(
            logits_row, filter_chain=filter_chain, is_greedy=is_greedy, temp=temp
        )

    # --- Step 1: bootstrap forward feeds the caller's pending token, and
    # samples y0 (the model's real, unconditional next continuation).
    y_in = mx.array([[last_committed_token_id]], dtype=mx.uint32)
    boot_logits, boot_hidden = model(
        y_in, cache=cache, return_hidden=True, n_confirmed=0
    )
    y0, y0_logprobs = sample(boot_logits[0, -1, :])

    token_ids = [last_committed_token_id, y0]
    logprobs = [last_committed_logprobs, y0_logprobs]
    from_draft = [False, False]

    # --- Step 2: draft up to `max_draft` tokens (drafting rank only), each
    # step conditioned on the SAME real backbone hidden state (hidden at y0)
    # and the growing next_token_ids/mtp_cache - see module docstring.
    draft_ids: list[int] = []
    draft_logprobs: list[mx.array] = []
    if max_draft > 0 and is_drafting_rank:
        hidden_at_y0 = boot_hidden[:, -1:, :]
        next_tok = mx.array([[y0]], dtype=mx.uint32)
        for _ in range(max_draft):
            draft_logits = model.mtp_forward(hidden_at_y0, next_tok, mtp_cache)
            row = draft_logits[0, -1, :]
            row_lp = row - mx.logsumexp(row, axis=-1, keepdims=True)
            d = int(mx.argmax(row_lp).item())
            draft_ids.append(d)
            draft_logprobs.append(row_lp)
            next_tok = mx.array([[d]], dtype=mx.uint32)

    k = max_draft if max_draft > 0 else 0
    local_draft_array = mx.array(draft_ids if draft_ids else [0] * k, dtype=mx.uint32)

    # --- Step 3: broadcast draft ids so every rank agrees on the sequence.
    if k > 0:
        draft_array = broadcast_draft_ids(
            local_draft_array, contribute=is_drafting_rank, group=group
        )
        drafted_token_ids = cast(list[int], draft_array.tolist())
    else:
        drafted_token_ids = []

    if not drafted_token_ids:
        return MtpStepResult(
            token_ids=token_ids,
            logprobs=logprobs,
            from_draft=from_draft,
            next_last_committed_token_id=y0,
            next_last_committed_logprobs=y0_logprobs,
            num_drafted=0,
            num_accepted=0,
        )

    # --- Step 4: one joint forward verifies the whole draft chunk.
    verify_in = mx.array([[y0] + drafted_token_ids], dtype=mx.uint32)
    verify_logits, _ = model(verify_in, cache=cache, return_hidden=True, n_confirmed=1)
    target_logprobs = [
        verify_logits[0, i, :] - mx.logsumexp(verify_logits[0, i, :], keepdims=True)
        for i in range(verify_logits.shape[1])
    ]

    uniform_draws = (
        [] if is_greedy else [float(mx.random.uniform().item()) for _ in range(k)]
    )
    result = sequential_accept_reject(
        drafted_token_ids,
        target_logprobs,
        is_greedy=is_greedy,
        uniform_draws=uniform_draws,
    )

    if result.num_accepted == k:
        # Every draft accepted: cache state is already correct, no rollback.
        assert result.bonus_token is not None
        for i in range(k):
            token_ids.append(drafted_token_ids[i])
            logprobs.append(draft_logprobs[i] if draft_logprobs else target_logprobs[i])
            from_draft.append(True)
        return MtpStepResult(
            token_ids=token_ids,
            logprobs=logprobs,
            from_draft=from_draft,
            next_last_committed_token_id=result.bonus_token,
            next_last_committed_logprobs=result.final_logprobs,
            num_drafted=k,
            num_accepted=k,
        )

    # Partial or zero accept: roll the whole draft chunk back and redo a
    # plain forward over [accepted prefix..., corrective] to re-populate
    # cache state (see rollback_mtp_cache's docstring for why).
    m = result.num_accepted
    assert result.corrective_token is not None
    rollback_mtp_cache(cache, k)
    redo_tokens = drafted_token_ids[:m] + [result.corrective_token]
    model(mx.array([redo_tokens], dtype=mx.uint32), cache=cache, n_confirmed=0)

    for i in range(m):
        token_ids.append(drafted_token_ids[i])
        logprobs.append(draft_logprobs[i] if draft_logprobs else target_logprobs[i])
        from_draft.append(True)

    return MtpStepResult(
        token_ids=token_ids,
        logprobs=logprobs,
        from_draft=from_draft,
        next_last_committed_token_id=result.corrective_token,
        next_last_committed_logprobs=result.final_logprobs,
        num_drafted=k,
        num_accepted=m,
    )


@dataclass
class MtpAcceptStats:
    """Running accept-rate / throughput counters, logged periodically."""

    steps: int = 0
    drafted: int = 0
    accepted: int = 0
    emitted: int = 0
    _log_every: int = field(default=50, repr=False)

    def record(self, result: MtpStepResult) -> None:
        self.steps += 1
        self.drafted += result.num_drafted
        self.accepted += result.num_accepted
        self.emitted += len(result.token_ids)

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted > 0 else 0.0

    def maybe_log(self) -> None:
        if self.steps > 0 and self.steps % self._log_every == 0:
            logger.info(
                f"MTP stats: steps={self.steps} accept_rate={self.accept_rate:.1%} "
                f"drafted={self.drafted} accepted={self.accepted} "
                f"tokens_emitted={self.emitted} "
                f"effective_tokens_per_step={self.emitted / self.steps:.2f}"
            )
