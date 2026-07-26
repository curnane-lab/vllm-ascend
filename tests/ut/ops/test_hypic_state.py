# SPDX-License-Identifier: Apache-2.0
"""Numerical validation of the Hypic state composition law.

The CPU tests validate the composition math against a token-level
reference unroll. The NPU test validates the Triton transition-extraction
kernel by linearity: h_recurrence is linear in the initial state, so for
any h0 it must hold that  S(h0) = T_C @ h0 + S(0).
"""

import pytest
import torch

from vllm_ascend.ops.hypic.state import SegmentCache, compose_states

try:
    import torch_npu  # noqa: F401

    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def _unroll(transitions: torch.Tensor, writes: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
    """Token-level reference: S_i = T_i @ S_{i-1} + u_i."""
    s = s0
    for t, u in zip(transitions, writes):
        s = torch.matmul(t, s) + u
    return s


def _make_segment(length, H, K, V, dtype, gdn_form=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    if gdn_form:
        # Dense-family form: T_t = diag(g_t) (I - beta_t k_t k_t^T)
        g = torch.rand(length, H, K, generator=gen, dtype=dtype) * 0.5 + 0.4
        beta = torch.rand(length, H, 1, generator=gen, dtype=dtype) * 0.3
        kk = torch.randn(length, H, K, generator=gen, dtype=dtype)
        kk = kk / kk.norm(dim=-1, keepdim=True)
        eye = torch.eye(K, dtype=dtype).expand(length, H, K, K)
        ts = g.unsqueeze(-1) * (eye - beta.unsqueeze(-1) * kk.unsqueeze(-1) @ kk.unsqueeze(-2))
    else:
        # Generic contractive transitions
        ts = torch.randn(length, H, K, K, generator=gen, dtype=dtype) * 0.1
        ts = ts + torch.eye(K, dtype=dtype).expand(length, H, K, K) * 0.5
    us = torch.randn(length, H, K, V, generator=gen, dtype=dtype) * 0.1
    return ts, us


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("gdn_form", [False, True])
def test_compose_matches_token_level_unroll(dtype, gdn_form):
    torch.manual_seed(0)
    H, K, V = 2, 8, 16
    seg_lens = [5, 3, 7]
    prefix = torch.randn(H, K, V, dtype=dtype)

    full_ts, full_us, seg_caches = [], [], []
    for i, seg_len in enumerate(seg_lens):
        ts, us = _make_segment(seg_len, H, K, V, dtype, gdn_form, seed=i)
        full_ts.append(ts)
        full_us.append(us)
        eye = torch.eye(K, dtype=dtype).expand(H, K, K)
        zero = torch.zeros(H, K, V, dtype=dtype)
        seg_caches.append(
            SegmentCache(
                transition=_unroll(ts, torch.zeros(seg_len, H, K, K, dtype=dtype), eye),
                zero_start_state=_unroll(ts, us, zero),
            )
        )

    # Reference: single-pass unroll of the whole sequence from `prefix`.
    ref = prefix
    for ts, us in zip(full_ts, full_us):
        ref = _unroll(ts, us, ref)

    got = compose_states(
        prefix,
        torch.stack([c.transition for c in seg_caches]),
        torch.stack([c.zero_start_state for c in seg_caches]),
    )

    tol = 1e-10 if dtype == torch.float64 else 1e-4
    rel = (got - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < tol, f"relative error {rel:.2e} exceeds {tol}"


def test_compose_zero_prefix_matches_zero_start():
    """Composing from a zero prefix must recover plain concatenation."""
    H, K, V = 2, 8, 8
    caches = []
    for i in range(3):
        ts, us = _make_segment(4, H, K, V, torch.float64, seed=10 + i)
        eye = torch.eye(K, dtype=torch.float64).expand(H, K, K)
        zero = torch.zeros(H, K, V, dtype=torch.float64)
        caches.append((_unroll(ts, torch.zeros(4, H, K, K, dtype=torch.float64), eye), _unroll(ts, us, zero), ts, us))

    ref = torch.zeros(H, K, V, dtype=torch.float64)
    for _, _, ts, us in caches:
        ref = _unroll(ts, us, ref)

    got = compose_states(
        torch.zeros(H, K, V, dtype=torch.float64),
        torch.stack([c[0] for c in caches]),
        torch.stack([c[1] for c in caches]),
    )
    assert (got - ref).norm() < 1e-10


@pytest.mark.skipif(not HAS_NPU, reason="torch_npu unavailable")
def test_transition_extraction_linearity_npu():
    """T_C extracted with (h0=I, u=0) must satisfy S(h0) = T_C @ h0 + S(0)."""
    from vllm_ascend.ops.hypic.transition import extract_segment_transitions
    from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    device = "npu"
    B, T, H, K = 1, 130, 2, 64  # spans 3 chunks of 64
    gen = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, generator=gen)
    w = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, generator=gen) * 0.1
    u = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, generator=gen) * 0.1
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=device, generator=gen))
    g = g.cumsum(dim=1).contiguous()  # stand-in for chunk_local_cumsum output

    t_c = extract_segment_transitions(k, w, g)
    assert t_c.shape == (B, H, K, K)

    h0 = torch.randn(B, H, K, K, device=device, dtype=torch.float32, generator=gen)
    zero = torch.zeros_like(h0)
    _, _, s_h0 = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True
    )
    _, _, s_0 = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=zero, output_final_state=True
    )

    rhs = torch.matmul(t_c, h0) + s_0
    rel = (s_h0 - rhs).norm() / rhs.norm().clamp_min(1e-12)
    assert rel < 2e-2, f"linearity check failed: rel err {rel:.2e}"  # bf16 tolerance
