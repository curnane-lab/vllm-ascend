# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.ops.gdn_pool_io import (
    indexed_pool_prefill,
    run_indexed_pool_prefill,
    single_sequence_pool_batch,
    supports_pool_io,
)

_POOL_ALIGNMENT_BYTES = 64


def _aligned_bf16_pool(rows: int, *inner: int) -> torch.Tensor:
    row_size = 1
    for dim in inner:
        row_size *= dim
    raw = torch.empty((rows * row_size) + _POOL_ALIGNMENT_BYTES, dtype=torch.bfloat16)
    element_size = raw.element_size()
    misaligned_bytes = raw.data_ptr() % _POOL_ALIGNMENT_BYTES
    shift_bytes = (_POOL_ALIGNMENT_BYTES - misaligned_bytes) % _POOL_ALIGNMENT_BYTES
    if shift_bytes % element_size != 0:
        raise AssertionError("bfloat16 storage is not element-aligned")
    shift = shift_bytes // element_size
    pool = raw[shift : shift + rows * row_size].view(rows, *inner)
    assert pool.data_ptr() % _POOL_ALIGNMENT_BYTES == 0
    return pool


def test_supports_pool_io_accepts_aligned_bf16_pool():
    pool = _aligned_bf16_pool(2, 2, 4, 8)
    assert (pool.stride(0) * pool.element_size()) % _POOL_ALIGNMENT_BYTES == 0
    indices = torch.tensor([0], dtype=torch.int32)
    has_initial_state = torch.tensor([True])
    assert supports_pool_io(pool, indices, has_initial_state, pool.device)


def test_supports_pool_io_rejects_unaligned_or_non_bf16_state():
    pool = _aligned_bf16_pool(2, 1, 1, 1)
    indices = torch.tensor([0], dtype=torch.int32)
    has_initial_state = torch.tensor([True])
    assert not supports_pool_io(pool, indices, has_initial_state, pool.device)

    fp32_pool = torch.zeros(2, 2, 4, 8)
    assert not supports_pool_io(fp32_pool, indices, has_initial_state, fp32_pool.device)
    assert not supports_pool_io(pool, indices.to(torch.int16), has_initial_state, pool.device)
    assert not supports_pool_io(pool, indices, has_initial_state.to(torch.int32), pool.device)


def test_single_sequence_pool_batch_matches_packed_prefill():
    query = torch.zeros(1, 6, 2, 8)
    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
    slots = torch.tensor([3], dtype=torch.int64)
    assert single_sequence_pool_batch(query, cu_seqlens, slots)
    assert not single_sequence_pool_batch(query, torch.tensor([0, 2, 6]), slots)
    assert not single_sequence_pool_batch(query, cu_seqlens, torch.tensor([3, 4]))


def test_indexed_pool_prefill_scatters_transposed_final_state(monkeypatch):
    pool = torch.zeros(4, 2, 3, 5)
    src = torch.tensor([1], dtype=torch.int32)
    dst = torch.tensor([2], dtype=torch.int32)
    has_initial_state = torch.tensor([True])
    q = torch.zeros(1, 2, 4, 5)
    k = torch.zeros_like(q)
    v = torch.zeros(1, 2, 4, 3)
    g = torch.zeros(1, 4, 2)
    beta = torch.zeros(1, 4, 2)
    scattered: dict[str, torch.Tensor] = {}

    def fake_gather(state, indices, mask, **kwargs):
        assert state is pool
        assert torch.equal(indices, src)
        assert torch.equal(mask, has_initial_state)
        return torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(1, 2, 3, 5)

    def fake_scatter(state, indices, source):
        assert state is pool
        scattered["indices"] = indices.clone()
        scattered["source"] = source.clone()

    def fake_fwd(query, key, value, gate, beta_tensor, **kwargs):
        initial_state = kwargs["initial_state"]
        assert initial_state.shape == (1, 2, 5, 3)
        assert kwargs["layout"] == "BNSD"
        assert kwargs["state_v_first"] is False
        assert kwargs["chunk_size"] == 64
        return torch.full((1, query.shape[2], 2, 3), 7), torch.ones(1, 2, 5, 3)

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.gather_ssm_states", fake_gather)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.scatter_ssm_states_", fake_scatter)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_ctypes_fused_fwd", lambda: fake_fwd)

    output, extra = indexed_pool_prefill(q, k, v, g, beta, pool, src, dst, has_initial_state, 0.5)

    assert extra is None
    assert output.shape == (1, 4, 2, 3)
    assert torch.equal(scattered["indices"], dst)
    expected = torch.ones(1, 2, 5, 3).transpose(-1, -2).contiguous()
    torch.testing.assert_close(scattered["source"], expected)


def test_run_indexed_pool_prefill_skips_missing_ctypes_op(monkeypatch):
    def missing():
        raise ImportError("ctypes fused prefill is unavailable")

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_ctypes_fused_fwd", missing)
    query = torch.zeros(1, 2, 1, 4)
    value = torch.zeros(1, 2, 1, 4)
    gate = torch.zeros(1, 2, 1)
    state = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    slots = torch.tensor([0], dtype=torch.int32)
    has_initial_state = torch.tensor([False])

    assert (
        run_indexed_pool_prefill(
            query,
            query,
            value,
            gate,
            gate,
            state,
            slots,
            has_initial_state,
            0.5,
        )
        is None
    )
