"""Strided slot-layout smoke: backing [64, 655360] with padding, view into it."""
import sys
from pathlib import Path
import torch
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW
from vllm_ascend.ops.gdn_lq_flush_ascendc import flush_pools_ascendc

DEVICE = "npu"
H, V, K = 32, 128, 128
gen = torch.Generator().manual_seed(5)
backing_t = torch.randn(64, 655360, generator=gen).to(DEVICE) * 0.05
backing_k = backing_t.clone()
state_t = backing_t[:, : H * V * K].view(64, H, V, K)
state_k = backing_k[:, : H * V * K].view(64, H, V, K)
print("slot stride bytes:", state_t.stride(0) * 4)
idx = torch.randint(1, 64, (8,), generator=gen).to(DEVICE)
pools_t = GDNLeapQuantPools(state_t)
pools_k = GDNLeapQuantPools(state_k)
pools_t.init_from_dense(state_t, idx)
pools_k.init_from_dense(state_k, idx)


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


for t in range(96):
    g = torch.Generator().manual_seed(2000 + t)
    kn = _norm(torch.randn(8, H, K, generator=g)).to(DEVICE)
    qn = _norm(torch.randn(8, H, K, generator=g)).to(DEVICE)
    vx = torch.randn(8, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    g_log = (torch.randn(8, H, generator=g) * 0.05).to(DEVICE)
    beta = torch.rand(8, H, generator=g).to(torch.bfloat16).to(DEVICE)
    pools_t.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
    pools_k.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
    pools_t.flush(idx)
    assert flush_pools_ascendc(pools_k, idx), "ascendc flush unavailable"
    if t < 3 or t % 16 == 15:
        torch.npu.synchronize()
        print(f"step {t} done", flush=True)
    if (t + 1) % WINDOW == 0:
        a = pools_t.decode_state(idx)
        b = pools_k.decode_state(idx)
        rel = ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-6)).item()
        print(f"step {t + 1}: state rel diff = {rel:.5f}")
        assert rel < 2e-2
print("STRIDED LAYOUT SMOKE PASS")
