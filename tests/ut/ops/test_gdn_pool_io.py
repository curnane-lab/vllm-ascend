# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.ops.gdn_pool_io import (
    indexed_pool_prefill,
    map_prefill_source_indices,
    prepare_gdn_pool_prefill_cow,
    run_indexed_pool_prefill,
    single_sequence_pool_batch,
    supports_pool_io,
)
from vllm_ascend.ops.gdn_pool_layout import (
    prepare_gdn_pool_prefill_metadata,
    prepare_pool_qkv,
    supports_fused_layout,
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

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.gdn_pool_prefill_ascendc", lambda **kwargs: None)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.gather_ssm_states", fake_gather)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.scatter_ssm_states_", fake_scatter)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_ctypes_fused_fwd", lambda: fake_fwd)

    output, extra = indexed_pool_prefill(q, k, v, g, beta, pool, src, dst, has_initial_state, 0.5)

    assert extra is None
    assert output.shape == (1, 4, 2, 3)
    assert torch.equal(scattered["indices"], dst)
    expected = torch.ones(1, 2, 5, 3).transpose(-1, -2).contiguous()
    torch.testing.assert_close(scattered["source"], expected)


def test_indexed_pool_prefill_prefers_native_op(monkeypatch):
    native = torch.zeros(1, 4, 2, 3)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.gdn_pool_prefill_ascendc", lambda **kwargs: native)

    def fail_gather(*args, **kwargs):
        raise AssertionError("compact gather should not run when the native op returns")

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.gather_ssm_states", fail_gather)
    pool = torch.zeros(2, 2, 3, 5)
    slots = torch.tensor([1], dtype=torch.int32)
    output, extra = indexed_pool_prefill(
        torch.zeros(1, 2, 4, 5),
        torch.zeros(1, 2, 4, 5),
        torch.zeros(1, 2, 4, 3),
        torch.zeros(1, 4, 2),
        torch.zeros(1, 4, 2),
        pool,
        slots,
        slots,
        torch.tensor([True]),
        0.5,
    )
    assert output is native
    assert extra is None


def test_opapi_candidates_keep_prototype_and_custom_opp(monkeypatch, tmp_path):
    from vllm_ascend.ops import gdn_pool_prefill_ascendc as binding

    prototype = tmp_path / "proto.so"
    custom_root = tmp_path / "custom"
    custom_lib = custom_root / "op_api" / "lib" / "libcust_opapi.so"
    prototype.write_bytes(b"")
    custom_lib.parent.mkdir(parents=True)
    custom_lib.write_bytes(b"")
    monkeypatch.setattr(binding, "_PROTOTYPE_OPAPI", str(prototype))
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(custom_root))

    found = binding._opapi_candidates()

    assert found == [str(prototype), str(custom_lib)]


def test_ascendc_pool_prefill_returns_none_without_library(monkeypatch):
    from vllm_ascend.ops import gdn_pool_prefill_ascendc as binding

    monkeypatch.setattr(binding, "_AVAILABLE", None)
    monkeypatch.setattr(binding, "_BINDING", None)
    monkeypatch.setattr(binding, "_opapi_candidates", lambda: [])
    monkeypatch.setattr(binding, "_capturing", lambda: False)
    slots = torch.tensor([0], dtype=torch.int32)
    pool = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    query = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
    value = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
    gate = torch.zeros(1, 2, 1)

    assert (
        binding.gdn_pool_prefill_ascendc(
            pool,
            pool,
            slots,
            slots,
            torch.tensor([False]),
            query,
            query,
            value,
            gate,
            gate.to(torch.bfloat16),
            0.5,
        )
        is None
    )
    assert binding._AVAILABLE is False


@pytest.mark.parametrize("failure", [RuntimeError("launch failed"), 17])
def test_ascendc_pool_prefill_does_not_fallback_after_launch_failure(monkeypatch, failure):
    from vllm_ascend.ops import gdn_pool_prefill_ascendc as binding

    def call(*args):
        if isinstance(failure, Exception):
            raise failure
        return failure

    runtime = SimpleNamespace(call=call)
    library = SimpleNamespace(
        aclnnGdnPoolPrefillGetWorkspaceSize=object(),
        aclnnGdnPoolPrefill=object(),
    )
    monkeypatch.setattr(binding, "_capturing", lambda: False)
    monkeypatch.setattr(binding, "_binding", lambda: (runtime, None, library))
    slots = torch.tensor([0], dtype=torch.int32)
    pool = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    query = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
    value = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
    gate = torch.zeros(1, 2, 1)

    with pytest.raises(RuntimeError, match="may be partially written"):
        binding.gdn_pool_prefill_ascendc(
            pool,
            pool,
            slots,
            slots,
            torch.tensor([False]),
            query,
            query,
            value,
            gate,
            gate.to(torch.bfloat16),
            0.5,
        )


def test_run_indexed_pool_prefill_skips_missing_ctypes_op(monkeypatch):
    def missing():
        raise ImportError("ctypes fused prefill is unavailable")

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.gdn_pool_prefill_ascendc", lambda **kwargs: None)
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


def test_map_prefill_source_indices_uses_original_destinations():
    indices = torch.tensor([1, 2, 3], dtype=torch.int64)
    copies = [
        KVCacheBlockCopy(src_block_id=2, dst_block_id=1),
        KVCacheBlockCopy(src_block_id=1, dst_block_id=2),
    ]

    mapped = map_prefill_source_indices(indices, copies)

    assert mapped.dtype == torch.int32
    assert mapped.tolist() == [2, 1, 3]
    assert indices.tolist() == [1, 2, 3]


def test_run_indexed_pool_prefill_uses_independent_slots(monkeypatch):
    calls = {}

    def fake_pool_op(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return torch.zeros(4, 1, 4)

    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op",
        lambda: fake_pool_op,
    )
    query = torch.ones(1, 4, 1, 4, dtype=torch.bfloat16)
    value = torch.ones_like(query)
    gate = torch.zeros(1, 4, 1)
    pool = torch.zeros(32, 1, 4, 4, dtype=torch.bfloat16)

    output = run_indexed_pool_prefill(
        query,
        query,
        value,
        gate,
        gate,
        pool,
        torch.tensor([5, 5], dtype=torch.int64),
        torch.tensor([True, False]),
        0.5,
        dst_slots=torch.tensor([17, 29], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    args = calls["args"]
    kwargs = calls["kwargs"]
    assert args[0].shape == (4, 1, 4)
    assert args[2].shape == (4, 1, 4)
    assert args[3] is pool
    assert args[4].dtype == torch.int32
    assert args[4].tolist() == [5, 5]
    assert args[5].tolist() == [17, 29]
    assert kwargs["actual_seq_lengths"].tolist() == [2, 2]
    assert kwargs["has_initial_state"].tolist() == [True, False]
    assert output.shape == (1, 4, 1, 4)


def test_native_pool_uses_prepared_running_slots(monkeypatch):
    calls = {}

    def fake_pool_op(*args, **kwargs):
        calls["args"] = args
        return torch.zeros(2, 1, 4)

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op", lambda: fake_pool_op)
    query = torch.ones(1, 2, 1, 4, dtype=torch.bfloat16)
    pool = torch.zeros(32, 1, 4, 4, dtype=torch.bfloat16)
    running_slots = torch.tensor([18], dtype=torch.int64)

    run_indexed_pool_prefill(
        query,
        query,
        query,
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 1),
        pool,
        running_slots,
        torch.tensor([True]),
        0.5,
        dst_slots=running_slots,
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
    )

    args = calls["args"]
    assert args[3] is pool
    assert args[4].tolist() == [18]
    assert args[5].tolist() == [18]


def test_prepare_gdn_pool_prefill_cow_retains_all_copies(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_GDN_POOL_IO", "1")
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op",
        lambda: object(),
    )

    class Builder:
        copies = ()

        def set_gdn_pool_prefill_cow(self, copies):
            self.copies = copies

    class Group:
        def __init__(self):
            self.builder = Builder()

        def get_metadata_builder(self, _):
            return self.builder

    group = Group()
    copy = KVCacheBlockCopy(src_block_id=1, dst_block_id=3)

    assert prepare_gdn_pool_prefill_cow([[group]], {}, [copy]) is None
    assert group.builder.copies == ()

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op", lambda: None)
    assert prepare_gdn_pool_prefill_cow([[group]], {}, [copy]) is None
    assert group.builder.copies == ()


def test_supports_pool_io_rejects_noncontiguous_destination_heads():
    storage = torch.zeros(2, 2, 4, 16, dtype=torch.bfloat16)
    pool = storage[:, :, :, :8]
    assert pool.stride(0) * pool.element_size() % _POOL_ALIGNMENT_BYTES == 0
    indices = torch.tensor([0], dtype=torch.int32)
    has_initial_state = torch.tensor([True])

    assert not supports_pool_io(pool, indices, has_initial_state, pool.device)


def _shared_qkv(tokens, dtype=torch.bfloat16):
    mixed = torch.zeros(64, tokens, 128, dtype=dtype)
    return tuple(part.transpose(0, 1).unsqueeze(0) for part in mixed.split((16, 16, 32), dim=0))


def test_fused_layout_requires_shared_bf16_head_dim_128():
    q, k, v = _shared_qkv(4)
    assert supports_fused_layout(q, k, v)
    assert supports_fused_layout(q, k, k)
    assert not supports_fused_layout(q, k, v.clone())

    q_fp32, k_fp32, v_fp32 = _shared_qkv(4, dtype=torch.float32)
    assert not supports_fused_layout(q_fp32, k_fp32, v_fp32)

    mixed = torch.zeros(64, 4, 64, dtype=torch.bfloat16)
    q_small, k_small, v_small = (part.transpose(0, 1).unsqueeze(0) for part in mixed.split((16, 16, 32), dim=0))
    assert not supports_fused_layout(q_small, k_small, v_small)


def test_prepare_pool_qkv_selects_shape_specific_path(monkeypatch):
    q, k, v = _shared_qkv(4)
    calls = []

    def fake_qkv(query, key, value):
        calls.append("qkv")
        return query.squeeze(0), key.squeeze(0), value.squeeze(0)

    def fake_qk(query, key, value):
        calls.append("qk")
        return query.squeeze(0), key.squeeze(0), value.squeeze(0)

    def fake_l2norm(value):
        calls.append("l2")
        return value

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_layout.fused_head_first_qkv", fake_qkv)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_layout.fused_head_first_qk", fake_qk)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_layout.l2norm_fwd", fake_l2norm)

    for rows in (1, 2, 3, 4, 5):
        query, key, value = prepare_pool_qkv(q, k, v, rows)
        assert query.shape == key.shape == (4, 16, 128)
        assert value.shape == (4, 32, 128)
    long_q, long_k, long_v = _shared_qkv(513)
    prepare_pool_qkv(long_q, long_k, long_v, 1)
    assert calls == ["qkv", "qkv", "qk", "qk", "l2", "l2", "l2"]


def test_run_indexed_pool_prefill_reuses_precomputed_metadata(monkeypatch):
    calls = {}

    def fake_pool_op(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return torch.zeros(4, 1, 4)

    def fake_prepare_qkv(q, k, v, rows):
        calls["rows"] = rows
        return q.squeeze(0), k.squeeze(0), v.squeeze(0)

    def fail_diff(*args, **kwargs):
        raise AssertionError("actual_seq_lengths must come from precomputed metadata")

    monkeypatch.setenv("VLLM_ASCEND_GDN_POOL_IO", "1")
    monkeypatch.setenv("VLLM_ASCEND_GDN_POOL_LAYOUT_FUSION", "1")
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op", lambda: fake_pool_op)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.prepare_pool_qkv", fake_prepare_qkv)
    metadata = prepare_gdn_pool_prefill_metadata(
        torch.tensor([5, 5], dtype=torch.int64),
        torch.tensor([17, 29], dtype=torch.int64),
        torch.tensor([True, False]),
        torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.torch.diff", fail_diff)
    query = torch.ones(1, 4, 1, 4, dtype=torch.bfloat16)
    value = torch.ones_like(query)
    gate = torch.zeros(1, 4, 1)
    pool = torch.zeros(32, 1, 4, 4, dtype=torch.bfloat16)

    output = run_indexed_pool_prefill(
        query,
        query,
        value,
        gate,
        gate,
        pool,
        torch.tensor([5, 5], dtype=torch.int64),
        torch.tensor([True, False]),
        0.5,
        dst_slots=torch.tensor([17, 29], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
        pool_metadata=metadata,
    )

    assert calls["rows"] == 2
    assert calls["args"][4].tolist() == [5, 5]
    assert calls["args"][5].tolist() == [17, 29]
    assert calls["kwargs"]["actual_seq_lengths"].tolist() == [2, 2]
    assert calls["kwargs"]["has_initial_state"].tolist() == [True, False]
    assert output.shape == (1, 4, 1, 4)


def test_layout_fusion_retains_original_pool_preparation_during_capture(monkeypatch):
    calls = {}

    def fake_pool_op(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return torch.zeros(4, 1, 4)

    def fail_prepare_qkv(*args, **kwargs):
        raise AssertionError("layout fusion must not run during graph capture")

    monkeypatch.setenv("VLLM_ASCEND_GDN_POOL_IO", "1")
    monkeypatch.setenv("VLLM_ASCEND_GDN_POOL_LAYOUT_FUSION", "1")
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._capturing", lambda: True)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op", lambda: fake_pool_op)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.prepare_pool_qkv", fail_prepare_qkv)
    monkeypatch.setattr("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", lambda value: value)
    query = torch.ones(1, 4, 1, 4, dtype=torch.bfloat16)
    value = torch.ones_like(query)
    gate = torch.zeros(1, 4, 1)
    pool = torch.zeros(32, 1, 4, 4, dtype=torch.bfloat16)

    output = run_indexed_pool_prefill(
        query,
        query,
        value,
        gate,
        gate,
        pool,
        torch.tensor([5, 5], dtype=torch.int64),
        torch.tensor([True, False]),
        0.5,
        dst_slots=torch.tensor([17, 29], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    assert calls["args"][4].tolist() == [5, 5]
    assert calls["args"][5].tolist() == [17, 29]
    assert calls["kwargs"]["actual_seq_lengths"].tolist() == [2, 2]
    assert output.shape == (1, 4, 1, 4)


def test_phase6_pool_reuses_host_metadata_and_storage(monkeypatch):
    calls = {}

    def fake_phase6_pool(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_phase6_pool_prefill_op", lambda: fake_phase6_pool)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op", lambda: None)
    monkeypatch.setattr("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", lambda value: value)
    monkeypatch.setattr(
        "vllm_ascend.ops.gdn_pool_io.torch.diff",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("host metadata must be reused")),
    )
    query = torch.ones(1, 4, 2, 8, dtype=torch.bfloat16)
    pool = torch.zeros(32, 2, 8, 8, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 4),
        chunk_indices_chunk64_host=(0, 0),
        keep_meta=None,
    )

    output = run_indexed_pool_prefill(
        query,
        query,
        query,
        torch.zeros(1, 4, 2),
        torch.zeros(1, 4, 2, dtype=torch.bfloat16),
        pool,
        torch.tensor([5], dtype=torch.int64),
        torch.tensor([True]),
        0.5,
        dst_slots=torch.tensor([17], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        chunk_metadata=metadata,
    )

    args = calls["args"]
    assert args[0].shape == (1, 2, 4, 8)
    assert not args[0].is_contiguous()
    assert args[0].data_ptr() == query.data_ptr()
    assert args[1].shape == args[0].shape
    assert args[2].shape == args[0].shape
    assert args[3].shape == (1, 4, 2)
    assert args[4].shape == (1, 4, 2)
    assert args[5] is pool
    assert args[6].tolist() == [5]
    assert args[7].tolist() == [17]
    assert calls["kwargs"]["cu_seqlens"] is metadata.cu_seqlens_host
    assert calls["kwargs"]["chunk_indices"] is metadata.chunk_indices_chunk64_host
    assert calls["kwargs"]["chunk_size"] == 64
    assert output.shape == (1, 4, 2, 8)


def test_phase6_pool_falls_back_to_legacy_for_filtered_sequences(monkeypatch):
    expected = torch.zeros(1, 2, 1, 4)
    fallback_calls = []

    def fail_phase6_pool(*args, **kwargs):
        raise AssertionError("filtered sequences must not use the Phase6 pool")

    def fake_legacy(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return expected, None

    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_phase6_pool_prefill_op", lambda: fail_phase6_pool)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io._get_pool_prefill_op", lambda: fail_phase6_pool)
    monkeypatch.setattr("vllm_ascend.ops.gdn_pool_io.indexed_pool_prefill", fake_legacy)
    monkeypatch.setattr("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", lambda value: value)
    query = torch.ones(1, 2, 1, 4, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 2),
        chunk_indices_chunk64_host=(0, 0),
        keep_meta=torch.tensor([0]),
    )

    output = run_indexed_pool_prefill(
        query,
        query,
        query,
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 1, dtype=torch.bfloat16),
        torch.zeros(8, 1, 4, 4, dtype=torch.bfloat16),
        torch.tensor([5], dtype=torch.int64),
        torch.tensor([True]),
        0.5,
        dst_slots=torch.tensor([17], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        chunk_metadata=metadata,
    )

    assert output is expected
    assert len(fallback_calls) == 1
