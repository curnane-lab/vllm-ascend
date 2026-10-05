"""Parity test: Triton-Ascend LeapQuant flush kernel vs eager torch pools flush."""

import sys

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW  # noqa: E402
from vllm_ascend.ops.triton.gdn_leapquant_flush import gdn_leapquant_flush  # noqa: E402

DEVICE = "npu"
B, H, HK, V, K, T = 8, 32, 16, 128, 128, 96
SLOTS = 64


def main():
    gen = torch.Generator().manual_seed(7)
    idx = torch.randint(1, SLOTS, (B,), generator=gen).to(DEVICE)
    buf_t = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    buf_k = buf_t.clone()
    pools_t = GDNLeapQuantPools(buf_t)
    pools_k = GDNLeapQuantPools(buf_k)
    pools_t.init_from_dense(buf_t, idx)
    pools_k.init_from_dense(buf_k, idx)

    ckpt_diffs = []
    for t in range(T):
        g = torch.Generator().manual_seed(1000 + t)
        kn = torch.randn(B, H, K, generator=g).to(torch.bfloat16).to(DEVICE)
        qn = torch.randn(B, H, K, generator=g).to(torch.bfloat16).to(DEVICE)
        vx = torch.randn(B, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
        g_log = (torch.randn(B, H, generator=g) * 0.05).to(DEVICE)
        beta = torch.rand(B, H, generator=g).to(torch.bfloat16).to(DEVICE)
        pools_t.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
        pools_k.decode_step(idx, kn, qn, vx, g_log, beta, 1.0)
        pools_t.flush(idx)
        gdn_leapquant_flush(pools_k, idx)
        if (t + 1) % WINDOW == 0:
            a = pools_t.decode_state(idx)
            b = pools_k.decode_state(idx)
            denom = a.abs().mean().clamp_min(1e-6)
            ckpt_diffs.append(((a - b).abs().mean() / denom).item())
    print(f"post-flush checkpoint rel diff (torch vs triton): {[f'{d:.5f}' for d in ckpt_diffs]}")
    print(f"max={max(ckpt_diffs):.5f}")
    assert max(ckpt_diffs) < 2e-2, "triton flush diverged"
    print("TRITON FLUSH PARITY PASS")


if __name__ == "__main__":
    main()
