"""M2-D2 stage 2 bench/parity: gdn_lq_gemv (AIC), chained after stage 1.

cq[:, :, :, 0] must match the v1 codesKn dot, [:, :, :, 1] the codesQn
dot, within fp16-input rounding (contract: rel L2 <= 3e-3 per column).
Timing runs BEFORE the reference computation; the fp32 reference is
computed on CPU in chunks (a torch-side advanced-indexing + einsum
combination at large slot counts faulted vector cores intermittently on
this box — unrelated to the custom ops, see results/M2-D2-stepv2.md).
"""

import sys
import time

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
from gdn_lq_dequant import gdn_lq_dequant, warm_cache as warm_dequant  # noqa: E402
from gdn_lq_gemv import gdn_lq_gemv, warm_cache as warm_gemv  # noqa: E402

DEVICE = "npu"
H, V, K = 32, 128, 128
SLOTS = 256
ITERS = 50

assert warm_dequant(DEVICE) and warm_gemv(DEVICE)


def run(b, iters=ITERS):
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(1, SLOTS, (b,), generator=gen).to(DEVICE)
    buf = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    g = torch.Generator().manual_seed(1)
    nk = H // 2
    kn = (torch.randn(b, nk, K, generator=g) / K ** 0.5).to(torch.bfloat16).to(DEVICE)
    qn = (torch.randn(b, nk, K, generator=g) / K ** 0.5).to(torch.bfloat16).to(DEVICE)
    scale = K ** -0.5

    staging = torch.zeros(b, H, V, K, dtype=torch.float16, device=DEVICE)
    aq = torch.zeros(b, H, 16, K, dtype=torch.float16, device=DEVICE)
    cq = torch.zeros(b, H, 16, V, dtype=torch.float32, device=DEVICE)
    ok = gdn_lq_dequant(pools._state, idx, kn, qn, staging, aq, H, nk, scale)
    assert ok, "stage1 unavailable"
    ok = gdn_lq_gemv(staging, aq, cq)
    if not ok:
        print(f"B={b}: stage2 unavailable")
        return None
    torch.npu.synchronize()

    gdn_lq_gemv(staging, aq, cq)  # warm
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        gdn_lq_gemv(staging, aq, cq)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3

    # chunked CPU reference against the actual fp16 operands. NOTE: torch
    # advanced indexing (x[grid1, grid2]) faults if launched after the
    # raw-aclnn custom ops on this stack (platform quirk, affects any
    # executor-launched op incl. the D1 probe) — gather on CPU instead.
    # staging/aq/cq are already [b, H, ...] (op outputs indexed by token)
    codes_f = staging.cpu().float()                             # [b,H,V,K]
    aq_c = aq.cpu().float()                                     # [b,H,16,K]
    cq_c = cq.cpu()                                             # [b,H,16,V]
    errs = []
    for c in range(2):
        num = 0.0
        den = 0.0
        for i0 in range(0, b):
            got = cq_c[i0, :, c, :]                        # [H,V]
            ref = torch.einsum("hvk,hk->hv", codes_f[i0],
                               aq_c[i0, :, c, :])          # [H,V]
            num += ((got - ref) ** 2).sum().item()
            den += (ref ** 2).sum().item()
        errs.append((num / max(den, 1e-30)) ** 0.5)
    return errs[0], errs[1], ms


print(f"{'B':>4} {'G':>6} {'relL2_kn':>12} {'relL2_qn':>12} {'ms':>9}")
all_ok = True
for b in (16, 64, 256):
    r = run(b)
    if r is None:
        sys.exit(1)
    err_k, err_q, ms = r
    ok = max(err_k, err_q) <= 3e-3
    all_ok &= ok
    print(f"{b:4d} {b * H:6d} {err_k:12.6f} {err_q:12.6f} {ms:9.3f}  "
          f"{'PASS' if ok else 'FAIL'}")
print("GEMV PARITY PASS" if all_ok else "GEMV PARITY FAIL")
sys.exit(0 if all_ok else 1)
