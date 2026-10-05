"""Parity test: Triton-Ascend LeapQuant step kernel vs the eager torch pools."""

import sys

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
from vllm_ascend.ops.triton.gdn_leapquant_step import gdn_leapquant_step  # noqa: E402

DEVICE = "npu"
B, H, HK, V, K, T = 8, 32, 16, 128, 128, 96
SLOTS = 64


def make_state(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    buf = torch.randn(SLOTS, H, V, K, generator=g).to(DEVICE) * 0.05
    return buf


def gen_inputs(gen):
    qx = torch.randn(B, HK, K, generator=gen).to(torch.bfloat16).to(DEVICE)
    kx = torch.randn(B, HK, K, generator=gen).to(torch.bfloat16).to(DEVICE)
    vx = torch.randn(B, H, V, generator=gen).to(torch.bfloat16).to(DEVICE)
    a = (torch.randn(B, H, generator=gen) * 2.0).to(torch.bfloat16).to(DEVICE)
    b = (torch.randn(B, H, generator=gen) * 2.0).to(torch.bfloat16).to(DEVICE)
    A_log = torch.log(torch.rand(H, generator=gen) * 0.09 + 0.01).to(DEVICE)
    dt_bias = (torch.randn(H, generator=gen) * 0.5).to(DEVICE)
    g = a.float() + dt_bias.view(1, H)
    sp = torch.where(g <= 20.0, torch.log1p(torch.exp(g)), g)
    at = torch.exp(-torch.exp(A_log).view(1, H) * sp)
    beta = torch.sigmoid(b.float()).to(torch.bfloat16)
    return qx, kx, vx, at, beta


def norm(x):
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)


def main():
    gen = torch.Generator().manual_seed(7)
    idx = torch.randint(1, SLOTS, (B,), generator=gen).to(DEVICE)
    # two identical state buffers: one driven by torch pools, one by triton
    buf_t = make_state(0)
    buf_k = buf_t.clone()
    pools_t = GDNLeapQuantPools(buf_t)
    pools_k = GDNLeapQuantPools(buf_k)
    pools_t.init_from_dense(buf_t, idx)
    pools_k.init_from_dense(buf_k, idx)

    errs = []
    for t in range(T):
        qx, kx, vx, at, beta = gen_inputs(gen)
        kn = norm(kx).repeat_interleave(H // HK, dim=1)
        qn = norm(qx).repeat_interleave(H // HK, dim=1)
        g_log = torch.log(at)
        o_t = pools_t.decode_step(idx, kn.bfloat16(), qn.bfloat16(), vx, g_log, beta, 1.0)
        o_k = gdn_leapquant_step(pools_k, idx, kn.bfloat16(), qn.bfloat16(), vx, g_log, beta, 1.0)
        pools_t.flush(idx)
        pools_k.flush(idx)
        denom = o_t.float().abs().mean().clamp_min(1e-6)
        errs.append(((o_k.float() - o_t.float()).abs().mean() / denom).item())
    print(f"triton-vs-torch output rel err: max={max(errs):.6f} last16 max={max(errs[-16:]):.6f}")
    # buffer bookkeeping parity
    ti, hi = pools_t.th_index(idx)
    for name in ["w", "p", "fill", "codes", "kbuf", "ubuf"]:
        a = getattr(pools_t, name)[ti, hi]
        b = getattr(pools_k, name)[ti, hi]
        same = torch.equal(a.float().cpu(), b.float().cpu())
        md = (a.float() - b.float()).abs().max().item()
        print(f"  {name}: bitexact={same} maxdiff={md:.6f}")
    assert max(errs) < 2e-2, "triton step diverged"
    print("TRITON STEP PARITY PASS")


if __name__ == "__main__":
    main()
