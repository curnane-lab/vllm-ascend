"""CPU reproduction of the M5 foundation contracts.

Validates the reconstructed dense-state APIs against the existing pool
semantics without requiring Ascend hardware:
1. quantize_from_tensor and init_from_dense agree bit-for-bit;
2. checkpoint recovery is bounded immediately after quantization;
3. reconstruct_dense tracks an fp32 recurrence while fill < WINDOW, and
   matches it again after flush folds the full window into the checkpoint;
4. external re-quantization invalidates the flush-v2 mirror epoch.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm-ascend"))

from vllm_ascend.ops.gdn_leapquant import (  # noqa: E402
    GDNLeapQuantPools,
    WINDOW,
)


def _make_inputs(slots, seed=11):
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(slots, 8, 128, 128, generator=generator) * 0.05
    idx = torch.arange(4)
    dense = state[idx].clone()
    inputs = {}
    key_count = 4
    key = torch.randn(4, key_count, 128, generator=generator)
    inputs["kn"] = (key / key.norm(dim=-1, keepdim=True)).to(torch.bfloat16).repeat_interleave(
        8 // key_count, dim=1
    )
    query = torch.randn(4, key_count, 128, generator=generator)
    inputs["qn"] = (query / query.norm(dim=-1, keepdim=True)).to(torch.bfloat16).repeat_interleave(
        8 // key_count, dim=1
    )
    inputs["v"] = torch.randn(4, 8, 128, generator=generator).to(torch.bfloat16)
    inputs["g_log"] = -(torch.rand(4, 8, generator=generator) * 0.1 + 0.005)
    inputs["beta"] = torch.rand(4, 8, generator=generator).to(torch.bfloat16)
    return state, idx, dense, inputs


def test_quantize_from_tensor_matches_init_from_dense():
    state, idx, dense, _ = _make_inputs(8)
    init_pools = GDNLeapQuantPools(state.clone())
    quant_pools = GDNLeapQuantPools(state.clone())

    init_pools.init_from_dense(state, idx)
    quant_pools.quantize_from_tensor(dense, idx)

    for field in ("codes", "s_k", "s_v", "u", "q", "w", "p", "fill", "kbuf", "ubuf"):
        assert torch.equal(
            getattr(init_pools, field)[idx], getattr(quant_pools, field)[idx]
        ), field


def test_checkpoint_round_trip():
    state, idx, dense, _ = _make_inputs(8, seed=12)
    pools = GDNLeapQuantPools(state)
    pools.init_from_dense(state, idx)
    recovered = pools.reconstruct_dense(idx)
    rel_error = ((recovered - dense).norm() / dense.norm()).item()
    assert rel_error < 0.02, rel_error


def _fp32_step(state, inputs):
    decay = torch.exp(inputs["g_log"].float()).unsqueeze(-1).unsqueeze(-1)
    decayed = decay * state
    key = inputs["kn"].float()
    update = inputs["beta"].float().unsqueeze(-1) * (
        inputs["v"].float() - torch.einsum("bhvk,bhk->bhv", decayed, key))
    return decayed + torch.einsum("bhv,bhk->bhvk", update, key)


def test_reconstruct_after_decode_and_flush():
    state, idx, _, inputs = _make_inputs(8, seed=13)
    pools = GDNLeapQuantPools(state)
    pools.init_from_dense(state, idx)
    reference = pools.reconstruct_dense(idx).clone()
    for step in range(WINDOW):
        reference = _fp32_step(reference, inputs)
        pools.decode_step(idx, **inputs, scale=1.0)
        reconstructed = pools.reconstruct_dense(idx)
        rel_error = ((reconstructed - reference).norm() / reference.norm()).item()
        if step + 1 < WINDOW:
            # fill < L: the bf16 window is part of the reconstruction.
            assert rel_error < 1e-2, (step + 1, rel_error)
        else:
            # fill == L is a new window, so reconstruct drops the buffer
            # until flush folds it. The gap here is the unfolded window.
            assert rel_error > 0.5, (step + 1, rel_error)
    pools.flush(idx)
    folded = pools.reconstruct_dense(idx)
    rel_error = ((folded - reference).norm() / reference.norm()).item()
    assert rel_error < 1e-2, rel_error
    assert torch.all(pools.fill[idx] == WINDOW)


def test_quantize_bumps_epoch_and_resets_window():
    state, idx, dense, inputs = _make_inputs(8, seed=14)
    pools = GDNLeapQuantPools(state)
    pools.init_from_dense(state, idx)
    epoch = pools._lq_epoch
    pools.decode_step(idx, **inputs, scale=1.0)
    assert torch.any(pools.fill[idx] == 1)

    pools.quantize_from_tensor(dense, idx)
    assert pools._lq_epoch == epoch + 1
    assert torch.all(pools.fill[idx] == 0)
    assert torch.count_nonzero(pools.kbuf[idx]) == 0
    assert torch.count_nonzero(pools.ubuf[idx]) == 0


if __name__ == "__main__":
    test_quantize_from_tensor_matches_init_from_dense()
    print("M5 CPU 1/4 PASS: quantize_from_tensor == init_from_dense")
    test_checkpoint_round_trip()
    print("M5 CPU 2/4 PASS: checkpoint round-trip")
    test_reconstruct_after_decode_and_flush()
    print("M5 CPU 3/4 PASS: reconstruct through decode and flush")
    test_quantize_bumps_epoch_and_resets_window()
    print("M5 CPU 4/4 PASS: epoch + window reset")
    print("M5 CPU FOUNDATIONS ALL PASS")
