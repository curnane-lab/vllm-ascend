"""Single-layer parity test: LeapQuant pools vs dense fp32 GDN recurrence (no vLLM).

Runs on NPU when available. Validates the exact math used by the vllm-ascend
hook before any full-model run.
"""

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW  # noqa: E402

DEVICE = "npu" if torch.npu.is_available() else "cpu"

B, H, HK, V, K, T = 4, 32, 16, 128, 128, 96


def gen_inputs(gen):
    qx = torch.randn(B, HK, K, generator=gen).to(torch.bfloat16).to(DEVICE)
    kx = torch.randn(B, HK, K, generator=gen).to(torch.bfloat16).to(DEVICE)
    vx = torch.randn(B, H, V, generator=gen).to(torch.bfloat16).to(DEVICE)
    a = (torch.randn(B, H, generator=gen) * 2.0).to(torch.bfloat16).to(DEVICE)
    b = (torch.randn(B, H, generator=gen) * 2.0).to(torch.bfloat16).to(DEVICE)
    A_log = torch.log(torch.rand(H, generator=gen) * 0.09 + 0.01).to(DEVICE)
    dt_bias = (torch.randn(H, generator=gen) * 0.5).to(DEVICE)
    return qx, kx, vx, a, b, A_log, dt_bias


def gating(a, b, A_log, dt_bias):
    """GDN scalar per-head decay: g = a + dt_bias (per head), softplus, at = exp(-exp(A_log)*sp)."""
    g = a.float() + dt_bias.view(1, H)
    sp = torch.where(g <= 20.0, torch.log1p(torch.exp(g)), g)
    at = torch.exp(-torch.exp(A_log).view(1, H) * sp)  # [B, H] in (0, 1]
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).float()
    return at, beta


def main():
    gen = torch.Generator().manual_seed(1234)
    state_buf = torch.zeros(B, H, V, K, device=DEVICE)
    pools = GDNLeapQuantPools(state_buf)
    S = torch.zeros(B, H, V, K, device=DEVICE)
    idx = torch.arange(B, device=DEVICE)
    errs = []
    state_errs = []
    for t in range(T):
        qx, kx, vx, a, b, A_log, dt_bias = gen_inputs(gen)
        at, beta = gating(a, b, A_log, dt_bias)
        rep = H // HK
        kf = kx.float().repeat_interleave(rep, 1)
        qf = qx.float().repeat_interleave(rep, 1)
        kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
        # dense oracle: decay first, then the delta-rule correction reads the
        # decayed state; the output uses the state after this token's update
        S_dec = at.view(B, H, 1, 1) * S
        u = beta.unsqueeze(-1) * (vx.float() - torch.einsum("bhvk,bhk->bhv", S_dec, kn))
        o_dense = torch.einsum("bhvk,bhk->bhv", S_dec, qn) + (kn * qn).sum(-1, keepdim=True) * u
        S = S_dec + torch.einsum("bhv,bhk->bhvk", u, kn)
        # leapquant path
        g_log = torch.log(at)
        o_lq = pools.decode_step(
            idx, kn.to(torch.bfloat16), qn.to(torch.bfloat16), vx, g_log, beta.to(torch.bfloat16), 1.0
        )
        pools.flush(idx)
        rel = (o_lq.float() - o_dense).abs().mean() / o_dense.abs().mean()
        errs.append(rel.item())
        if (t + 1) % WINDOW == 0:
            state_err = (pools.decode_state(idx) - S).abs().mean() / S.abs().mean().clamp_min(1e-6)
            state_errs.append(state_err.item())
    print(f"output rel err: first16 max={max(errs[:16]):.5f} "
          f"last16 max={max(errs[-16:]):.5f} overall max={max(errs):.5f}")
    print(f"post-flush state rel err per window: {[f'{e:.5f}' for e in state_errs]}")
    assert max(errs[:16]) < 2e-2, "first-window output error too large"
    assert max(errs[-16:]) < 5e-2, "error drifted too much over 6 windows"
    assert max(state_errs) < 2e-1, "state reconstruction error too large"
    print("PARITY PASS")


if __name__ == "__main__":
    main()
