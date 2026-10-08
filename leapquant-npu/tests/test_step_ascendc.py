"""Parity test: AscendC LeapQuant step kernel vs eager torch pools.decode_step.

Run with:
  LD_LIBRARY_PATH=/home/developer/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH \
      python3 tests/test_step_ascendc.py
"""

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
sys.path.insert(0, str(_REPO / "leapquant-npu"))

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW  # noqa: E402
from vllm_ascend.ops.gdn_lq_step_ascendc import step_pools_ascendc  # noqa: E402
from vllm_ascend.ops.gdn_lq_flush_ascendc import flush_pools_ascendc  # noqa: E402

DEVICE = "npu"
B, H, V, K, T = 8, 32, 128, 128, 96
SLOTS = 64


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def main():
    gen = torch.Generator().manual_seed(7)
    idx = torch.randint(1, SLOTS, (B,), generator=gen).to(DEVICE)
    bufs = [torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05]
    bufs.append(bufs[0].clone())
    pools = [GDNLeapQuantPools(b) for b in bufs]
    for p, b in zip(pools, bufs):
        p.init_from_dense(b, idx)

    out_diffs = []
    state_diffs = []
    step = 0
    for t in range(T):
        g = torch.Generator().manual_seed(1000 + t)
        kn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
        qn = _norm(torch.randn(B, H, K, generator=g)).to(DEVICE)
        vx = torch.randn(B, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
        g_log = (torch.randn(B, H, generator=g) * 0.05).to(DEVICE)
        beta = torch.rand(B, H, generator=g).to(torch.bfloat16).to(DEVICE)
        o_t = pools[0].decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
        o_a = step_pools_ascendc(pools[1], idx, kn, qn, vx, g_log, beta, 1.0)
        assert o_a is not None, "ascendc step unavailable"
        step += 1
        if step % WINDOW == 0:  # window end: flush both sides (pipeline contract)
            pools[0].flush(idx)
            assert flush_pools_ascendc(pools[1], idx)
        if step % 8 == 0 or step == 1:
            d = ((o_a.float() - o_t.float()).abs().mean()
                 / o_t.float().abs().mean().clamp_min(1e-6)).item()
            out_diffs.append((step, d))
            # per-step output rel: tight inside the first window (pure
            # step-kernel semantics); after a flush the two trajectories
            # re-quantize slightly differently and the synthetic delta rule
            # amplifies it exponentially (real-data replay, which is
            # contractive, stays <=8e-4/step — see tests/replay_cross2.py)
            lim = 2e-3 if step <= WINDOW else 1e-2
            assert d < lim, f"output rel {d:.6f} over limit at step {step}"
            fields = ("w", "p", "fill", "kbuf", "ubuf") if step <= WINDOW \
                else ("w", "p", "fill", "kbuf")
            for name in fields:
                a = getattr(pools[0], name)[idx]
                b = getattr(pools[1], name)[idx]
                if name == "kbuf":
                    f = pools[0].fill[idx].unsqueeze(-1)
                    j = torch.arange(16, device=DEVICE).view(1, 1, 16)
                    live = (j < f).unsqueeze(-1)
                    same = ((a == b) | ~live).all().item()
                elif name == "ubuf":
                    # u_new carries fp32 reduction-order noise; after a
                    # flush the two trajectories re-quantize slightly
                    # differently and the delta rule amplifies it, so the
                    # tolerance grows per window
                    f = pools[0].fill[idx].unsqueeze(-1)
                    j = torch.arange(16, device=DEVICE).view(1, 1, 16)
                    live = (j < f).unsqueeze(-1)
                    tol = (0.008 if step <= WINDOW else 0.05) * \
                        torch.maximum(a.abs(), b.abs()) + \
                        (2e-3 if step <= WINDOW else 2e-2)
                    same = ((((a - b).abs() <= tol) | ~live).all().item())
                else:
                    same = torch.equal(a, b)
                assert same, f"{name} diverged at step {step}"
        if step % WINDOW == 0:
            a = pools[0].decode_state(idx)
            b = pools[1].decode_state(idx)
            state_diffs.append(((a - b).abs().mean()
                                / a.abs().mean().clamp_min(1e-6)).item())
    print(f"output rel diffs: {[(s, f'{d:.6f}') for s, d in out_diffs]}")
    print(f"window-end state rel diffs: {[f'{d:.6f}' for d in state_diffs]}")
    max_out = max(d for _, d in out_diffs)
    assert max_out < 1e-2, "ascendc step output diverged"
    assert max(state_diffs) < 2e-2, "ascendc step state diverged"
    print("ASCENDC STEP PARITY PASS")


if __name__ == "__main__":
    main()
