"""Parity test: AscendC LeapQuant flush kernel vs eager torch pools flush.

Run with:
  LD_LIBRARY_PATH=/home/developer/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH \
      python3 tests/test_flush_ascendc.py
"""

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
sys.path.insert(0, str(_REPO / "leapquant-npu"))

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW  # noqa: E402
from vllm_ascend.ops.gdn_lq_flush_ascendc import (  # noqa: E402
    flush_pools_ascendc,
    gdn_lq_flush_ascendc,
)

DEVICE = "npu"
B, H, V, K, T = 8, 32, 128, 128, 96
SLOTS = 64


def _norm(x):
    # decode contract: kn/qn are l2-normalized bf16 (unnormalized keys make
    # the delta-rule recurrence diverge and overflow the fp16 comp tokens)
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def main():
    gen = torch.Generator().manual_seed(7)
    idx = torch.randint(1, SLOTS, (B,), generator=gen).to(DEVICE)
    buf_t = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    buf_k = buf_t.clone()
    pools_t = GDNLeapQuantPools(buf_t)
    pools_k = GDNLeapQuantPools(buf_k)
    pools_t.init_from_dense(buf_t, idx)
    pools_k.init_from_dense(buf_k, idx)

    # smoke: single flush on identical fresh pools, field-by-field check
    kn = _norm(torch.randn(B, H, K, generator=gen)).to(DEVICE)
    qn = _norm(torch.randn(B, H, K, generator=gen)).to(DEVICE)
    vx = torch.randn(B, H, V, generator=gen).to(torch.bfloat16).to(DEVICE)
    g_log = (torch.randn(B, H, generator=gen) * 0.05).to(DEVICE)
    beta = torch.rand(B, H, generator=gen).to(torch.bfloat16).to(DEVICE)
    for _ in range(WINDOW):
        pools_t.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
        pools_k.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
    ok = flush_pools_ascendc(pools_k, idx)
    print(f"ascendc flush available: {ok}")
    if not ok:
        raise SystemExit("ascendc flush unavailable")
    pools_t.flush(idx)
    torch.npu.synchronize()
    for name in ("codes", "s_k", "s_v", "u", "q", "w", "p", "fill"):
        a = getattr(pools_t, name)[idx]
        b = getattr(pools_k, name)[idx]
        if name == "codes":
            diff = (a != b).float().mean().item()
            print(f"  {name}: mismatch frac={diff:.6f}")
        else:
            d = (a.float() - b.float()).abs().mean() / a.float().abs().mean().clamp_min(1e-6)
            print(f"  {name}: rel diff={d.item():.6f}")

    # full parity run: 96 decode steps, compare checkpoint at window ends
    buf_t = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    buf_k = buf_t.clone()
    pools_t = GDNLeapQuantPools(buf_t)
    pools_k = GDNLeapQuantPools(buf_k)
    pools_t.init_from_dense(buf_t, idx)
    pools_k.init_from_dense(buf_k, idx)

    ckpt_diffs = []
    for t in range(T):
        g = torch.Generator().manual_seed(1000 + t)
        kn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
        qn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
        vx = torch.randn(B, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
        g_log = (torch.randn(B, H, generator=g) * 0.05).to(DEVICE)
        beta = torch.rand(B, H, generator=g).to(torch.bfloat16).to(DEVICE)
        pools_t.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
        pools_k.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
        pools_t.flush(idx)
        assert flush_pools_ascendc(pools_k, idx)
        if (t + 1) % WINDOW == 0:
            a = pools_t.decode_state(idx)
            b = pools_k.decode_state(idx)
            denom = a.abs().mean().clamp_min(1e-6)
            ckpt_diffs.append(((a - b).abs().mean() / denom).item())
    print(f"post-flush checkpoint rel diff (torch vs ascendc): "
          f"{[f'{d:.5f}' for d in ckpt_diffs]}")
    print(f"max={max(ckpt_diffs):.5f}")
    assert max(ckpt_diffs) < 2e-2, "ascendc flush diverged"
    print("ASCENDC FLUSH PARITY PASS")


if __name__ == "__main__":
    main()
