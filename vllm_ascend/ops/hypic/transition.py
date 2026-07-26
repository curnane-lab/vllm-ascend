# SPDX-License-Identifier: Apache-2.0
"""Extraction of the segment-cumulative transition operator T_C (Hypic).

The chunked GDN h-recurrence is linear in the incoming state:

    h_out = T_chunk @ h_in + (writes of the chunk's own tokens)

where the effective transition is fully determined by (k, w, g) — none of
which depend on v. Re-running the recurrence with an identity initial
state and the write term zeroed therefore yields exactly the
segment-cumulative transition operator of arXiv:2607.01299:

    T_C = prod_{t in C} T_t  =  h_recurrence(k, w, u=0, h0=I)

This mirrors the "second FLA invocation with S0 = I and u_t zeroed"
described in the paper's implementation section. Requires NPU (Triton).
"""

import torch

from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

_MAX_SUPPORTED_HEAD_DIM = 128  # the h-kernel splits V into two 64-wide blocks


def extract_segment_transitions(
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the per-segment transition operator T_C for a varlen batch.

    Args:
        k: keys of shape ``[B, T, Hg, K]`` (B = 1 when ``cu_seqlens`` is set).
        w: the WY-representation decay/gating term of shape ``[B, T, H, K]``,
            as produced by ``recompute_w_u_fwd``. Independent of v.
        g: chunk-local cumulative log-decay of shape ``[B, T, H]``, i.e. the
            output of ``chunk_local_cumsum``.
        cu_seqlens: cumulative segment lengths ``[N + 1]``; each entry range
            is one cacheable segment.
        chunk_indices / chunk_offsets: optional precomputed chunk metadata.

    Returns:
        T_C of shape ``[N, H, K, K]`` in float32, one transition operator
        per segment. ``T_C @ S + S_{C|0}`` composes the segment behind any
        prefix state S (see ``hypic.state.compose_states``).
    """
    if k.dim() != 4:
        raise ValueError(f"expected k of shape [B, T, Hg, K], got {k.shape}")
    B, T, Hg, K = k.shape
    if w.shape[:2] != (B, T) or w.shape[-1] != K:
        raise ValueError(f"w shape {w.shape} inconsistent with k shape {k.shape}")
    if g.shape[:2] != (B, T):
        raise ValueError(f"g shape {g.shape} inconsistent with k shape {k.shape}")
    H = w.shape[-2]
    if K > _MAX_SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"head dim K={K} exceeds the h-kernel limit of "
            f"{_MAX_SUPPORTED_HEAD_DIM} for transition extraction"
        )
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("varlen mode requires flattened input with B == 1")
        n_seq = len(cu_seqlens) - 1
    else:
        n_seq = B

    # Transition mode: the value/write channel becomes the K-dimensional
    # identity probe, so V := K and u := 0.
    u_zero = k.new_zeros(B, k.shape[1], H, K)
    eye = (
        torch.eye(K, dtype=torch.float32, device=k.device)
        .view(1, 1, K, K)
        .expand(n_seq, H, K, K)
        .contiguous()
    )

    _, _, transition = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u_zero,
        g=g,
        initial_state=eye,
        output_final_state=True,
        save_new_value=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    return transition
