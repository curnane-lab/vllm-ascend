# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, HEAD_BYTES, WINDOW


def _dense_state(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(128, 4, generator=generator)
    right = torch.randn(4, 128, generator=generator)
    return 0.1 * (left @ right) + 0.005 * torch.randn(128, 128, generator=generator)


@pytest.mark.parametrize("padded", [False, True])
def test_checkpoint_layout_and_reconstruction(padded):
    if padded:
        storage = torch.zeros(2, 128 * 128 + 64, dtype=torch.float32)
        state = storage[:, : 128 * 128].view(2, 1, 128, 128)
    else:
        state = torch.zeros(2, 1, 128, 128, dtype=torch.float32)
    dense = _dense_state()
    state[1, 0] = dense
    pools = GDNLeapQuantPools(state)

    pools.init_from_dense(state, torch.tensor([1]))

    assert pools.codes.dtype == torch.int8
    assert pools.u.dtype == torch.float16
    assert pools.q.dtype == torch.float16
    assert pools.s_k.dtype == torch.float32
    assert pools.s_v.dtype == torch.float32
    assert pools.kbuf.shape[1:] == (1, WINDOW, 128)
    assert pools.ubuf.shape[1:] == (1, WINDOW, 128)
    assert pools.codes.stride(0) == state.stride(0) * 4
    assert pools.ubuf.stride(0) == state.stride(0) * 2
    assert pools.p.stride(0) == state.stride(0)
    torch.testing.assert_close(pools.decode_state(torch.tensor([1]))[0, 0], dense, rtol=0.1, atol=0.02)


def test_compact_pool_quantize_and_reconstruct_dense():
    state = torch.zeros(2, 1, HEAD_BYTES // 4, dtype=torch.float32)
    pools = GDNLeapQuantPools(state)
    dense = _dense_state(3).unsqueeze(0).unsqueeze(0)
    idx = torch.tensor([1])

    pools.quantize_from_tensor(dense, idx)

    assert pools.head_v == 128
    assert pools.head_k == 128
    assert pools.codes.stride(0) == state.stride(0) * 4
    torch.testing.assert_close(pools.reconstruct_dense(idx)[0, 0], dense[0, 0], rtol=0.1, atol=0.02)


@pytest.mark.parametrize("kind", ["zero", "low_rank", "tiny"])
def test_quantization_of_degenerate_states_stays_finite(kind):
    state = torch.zeros(1, 1, 128, 128, dtype=torch.float32)
    if kind == "low_rank":
        state[0, 0] = torch.outer(torch.arange(128), torch.arange(128)).float()
    elif kind == "tiny":
        state[0, 0] = torch.outer(torch.arange(128), torch.arange(128)).float() * 1e-30
    pools = GDNLeapQuantPools(state)

    pools.init_from_dense(state, torch.tensor([0]))

    for tensor in (pools.codes, pools.s_k, pools.s_v, pools.u, pools.q, pools.decode_state(torch.tensor([0]))):
        assert torch.isfinite(tensor).all()


def test_decode_and_flush_match_dense_reference_across_windows():
    torch.manual_seed(7)
    state = torch.zeros(1, 1, 128, 128, dtype=torch.float32)
    state[0, 0] = _dense_state(7)
    reference = state[0, 0].clone()
    pools = GDNLeapQuantPools(state)
    idx = torch.tensor([0])
    pools.init_from_dense(state, idx)

    for step in range(WINDOW + 1):
        key = F.normalize(torch.randn(1, 1, 128), dim=-1).to(torch.bfloat16)
        query = F.normalize(torch.randn(1, 1, 128), dim=-1).to(torch.bfloat16)
        value = torch.randn(1, 1, 128).to(torch.bfloat16)
        gate = torch.full((1, 1), -0.05)
        beta = torch.full((1, 1), 0.3).to(torch.bfloat16)
        scale = 128**-0.5

        actual = pools.decode_step(idx, key, query, value, gate, beta, scale)
        pools.flush(idx)

        reference *= gate.exp()[0, 0]
        update = beta.float()[0, 0] * (value.float()[0, 0] - reference @ key.float()[0, 0])
        reference += torch.outer(update, key.float()[0, 0])
        expected = reference @ (query.float()[0, 0] * scale)
        torch.testing.assert_close(actual[0, 0].float(), expected, rtol=0.15, atol=0.04)

        if step == WINDOW - 1:
            assert pools.fill[0, 0] == WINDOW
            assert pools.p[0, 0] == 1
            assert torch.count_nonzero(pools.w[0, 0]) == 0
            torch.testing.assert_close(pools.decode_state(idx)[0, 0], reference, rtol=0.15, atol=0.04)

    assert pools.fill[0, 0] == 1


def test_reinitializing_recycled_slot_clears_window_state():
    state = torch.zeros(2, 1, 128, 128, dtype=torch.float32)
    pools = GDNLeapQuantPools(state)
    idx = torch.tensor([1])
    state[1, 0] = _dense_state(1)
    pools.init_from_dense(state, idx)
    key = F.normalize(torch.randn(1, 1, 128), dim=-1).to(torch.bfloat16)
    query = F.normalize(torch.randn(1, 1, 128), dim=-1).to(torch.bfloat16)
    value = torch.randn(1, 1, 128).to(torch.bfloat16)
    pools.decode_step(idx, key, query, value, torch.zeros(1, 1), torch.ones(1, 1).to(torch.bfloat16), 1.0)

    dense = _dense_state(2)
    state[1, 0] = dense
    pools.init_from_dense(state, idx)

    assert pools.fill[1, 0] == 0
    assert pools.p[1, 0] == 1
    assert torch.count_nonzero(pools.w[1, 0]) == 0
    assert torch.count_nonzero(pools.kbuf[1, 0]) == 0
    assert torch.count_nonzero(pools.ubuf[1, 0]) == 0
    torch.testing.assert_close(pools.decode_state(idx)[0, 0], dense, rtol=0.1, atol=0.02)


@pytest.mark.parametrize("padded", [False, True])
def test_triton_step_matches_torch_reference_on_npu(padded):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    from vllm_ascend.ops.triton.gdn_leapquant_step import gdn_leapquant_step

    dense = _dense_state(9).npu()
    if padded:
        storage = torch.zeros(2, 128 * 128 + 64, dtype=torch.float32, device="npu")
        kernel_state = storage[:, : 128 * 128].view(2, 1, 128, 128)
        reference_state = torch.zeros_like(kernel_state)
        idx = torch.tensor([1], device="npu")
    else:
        kernel_state = dense.unsqueeze(0).unsqueeze(0).clone()
        reference_state = dense.unsqueeze(0).unsqueeze(0).clone()
        idx = torch.tensor([0], device="npu")
    kernel_state[idx] = dense
    reference_state[idx] = dense
    kernel_pools = GDNLeapQuantPools(kernel_state)
    reference_pools = GDNLeapQuantPools(reference_state)
    kernel_pools.init_from_dense(kernel_state, idx)
    reference_pools.init_from_dense(reference_state, idx)

    for step in range(WINDOW + 1):
        key = F.normalize(torch.randn(1, 1, 128), dim=-1).to(torch.bfloat16).npu()
        query = F.normalize(torch.randn(1, 1, 128), dim=-1).to(torch.bfloat16).npu()
        value = torch.randn(1, 1, 128).to(torch.bfloat16).npu()
        gate = torch.full((1, 1), -0.05, device="npu")
        beta = torch.full((1, 1), 0.3, device="npu").to(torch.bfloat16)
        scale = 128**-0.5

        actual = gdn_leapquant_step(kernel_pools, idx, key, query, value, gate, beta, scale)
        expected = reference_pools.decode_step(idx, key, query, value, gate, beta, scale)
        kernel_pools.flush(idx)
        reference_pools.flush(idx)

        torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.05)
        assert torch.equal(kernel_pools.fill, reference_pools.fill)
        torch.testing.assert_close(kernel_pools.p, reference_pools.p)
        torch.testing.assert_close(kernel_pools.w, reference_pools.w, rtol=0.01, atol=1e-6)
        torch.testing.assert_close(kernel_pools.kbuf, reference_pools.kbuf)
        torch.testing.assert_close(kernel_pools.ubuf, reference_pools.ubuf, rtol=0.05, atol=0.02)
