"""M2-D2 stage 1 bench/parity: gdn_lq_dequant (AIV) vs torch reference.

staging must equal fp16(pools.codes) BIT-EXACT (pure int8->fp16 cast);
aq rows 0/1 = fp16(kn/qn expanded * s_k (*scale for qn) / 127), rows 2:15 = 0.
Timing caliber: warmup once (executor cache) -> sync -> perf_counter loop.
"""

import sys
import time

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, QMAX  # noqa: E402
from gdn_lq_dequant import gdn_lq_dequant  # noqa: E402

DEVICE = "npu"
H, V, K = 32, 128, 128
SLOTS = 2048
ITERS = 50


def make_pools(b, gen):
    idx = torch.randint(1, SLOTS, (b,), generator=gen).to(DEVICE)
    buf = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    return pools, idx


def run(b, iters=ITERS):
    gen = torch.Generator().manual_seed(0)
    pools, idx = make_pools(b, gen)
    g = torch.Generator().manual_seed(1)
    nk = H // 2
    kn = (torch.randn(b, nk, K, generator=g) / K ** 0.5).to(torch.bfloat16).to(DEVICE)
    qn = (torch.randn(b, nk, K, generator=g) / K ** 0.5).to(torch.bfloat16).to(DEVICE)
    scale = K ** -0.5

    staging = torch.zeros(b, H, V, K, dtype=torch.float16, device=DEVICE)
    aq = torch.zeros(b, H, 16, K, dtype=torch.float16, device=DEVICE)
    ok = gdn_lq_dequant(pools._state, idx, kn, qn, staging, aq, H, nk, scale)
    if not ok:
        print(f"B={b}: op unavailable")
        return None
    torch.npu.synchronize()

    # parity
    t_idx, h_idx = pools.th_index(idx)
    codes_ref = pools.codes[t_idx, h_idx]  # [b,H,V,K] int8
    staging_ok = torch.equal(staging.view(torch.int16),
                             codes_ref.to(torch.float16).view(torch.int16))
    sk = pools.s_k[t_idx, h_idx]           # [b,H,K]
    rep = H // nk
    kn_exp = kn.repeat_interleave(rep, dim=1).float()
    qn_exp = qn.repeat_interleave(rep, dim=1).float()
    kref = (kn_exp * sk / QMAX).to(torch.float16)
    qref = (qn_exp * sk * scale / QMAX).to(torch.float16)
    # aq: fp32 premultiply then one fp16 rounding; constant-folding order
    # differs from torch by <=1 ulp on ~1e-5 of elements — allclose is the
    # contract (staging above is the bit-exact part)
    aq_close = torch.allclose(aq[:, :, :2], torch.stack([kref, qref], dim=2),
                              rtol=1e-3, atol=1e-6)
    aq_err = max(
        (aq[:, :, 0, :].float() - kref.float()).abs().max().item(),
        (aq[:, :, 1, :].float() - qref.float()).abs().max().item(),
    )
    pad_zero = (aq[:, :, 2:, :] == 0).all().item()

    gdn_lq_dequant(pools._state, idx, kn, qn, staging, aq, H, nk, scale)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        gdn_lq_dequant(pools._state, idx, kn, qn, staging, aq, H, nk, scale)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return staging_ok, aq_close, aq_err, pad_zero, ms


print(f"{'B':>4} {'G':>6} {'staging_bitexact':>16} {'aq_maxerr':>10} "
      f"{'pad0':>5} {'ms':>9}")
all_ok = True
for b in (16, 64, 256):
    r = run(b)
    if r is None:
        sys.exit(1)
    staging_ok, aq_close, aq_err, pad_zero, ms = r
    ok = staging_ok and aq_close and pad_zero
    all_ok &= ok
    print(f"{b:4d} {b * H:6d} {str(staging_ok):>16} {aq_err:10.6f} "
          f"{str(pad_zero):>5} {ms:9.3f}  {'PASS' if ok else 'FAIL'}")
print("DEQUANT PARITY PASS" if all_ok else "DEQUANT PARITY FAIL")
sys.exit(0 if all_ok else 1)
