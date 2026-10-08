"""M2-D1 probe bench: cube-GEMV LeapQuant core vs torch fp32 reference.

Parity:  o[g,n,v] = sum_k (codes[g,k,v]*scale[g,k]) * q[g,n,k]
  codes int8 [G,128,128] (k rows, v cols), scale fp32 [G,128],
  q fp16 [G,N,128] (normalized rows), o fp32 [G,N,128].
Cube path: A = fp16(fp32(q)*scale) padded to [G,16,128], B = fp16(codes)
(exact), C = A@B on the cube (fp16 x fp16 -> fp32), o = C[:, :N, :].
Pass bar: relative L2 error <= 3e-3 for N = 1, 2, 8.

Timing caliber mirrors bench_step_ascendc.py: one warmup call (builds the
executor cache), torch.npu.synchronize, perf_counter loop, final sync.
"""

import sys
from pathlib import Path
import time

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "leapquant-npu"))

from gemv_probe import gdn_lq_gemv_probe  # noqa: E402

DEVICE = "npu"
K = 128
V = 128
M_PAD = 16
ITERS = 50


def make_data(g, n):
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(-127, 128, (g, K, V), generator=gen,
                          dtype=torch.int8)
    scale = (torch.rand(g, K, generator=gen) * 0.05 + 0.005)
    q = torch.randn(g, n, K, generator=gen)
    q = q / q.norm(dim=-1, keepdim=True)
    return (codes.to(DEVICE), scale.to(DEVICE),
            q.to(torch.float16).to(DEVICE))


def make_tiles(codes, scale, q):
    g, n = q.shape[0], q.shape[1]
    a = torch.zeros(g, M_PAD, K, dtype=torch.float32, device=q.device)
    a[:, :n, :] = q.float() * scale.unsqueeze(1)
    a = a.to(torch.float16)
    b = codes.to(torch.float16)
    return a, b


def reference(codes, scale, q):
    codes_f = codes.float() * scale.unsqueeze(-1)  # [G,K,V]
    return q.float() @ codes_f  # [G,N,V]


def rel_l2(o, ref):
    return ((o - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def run(g, n, iters=ITERS):
    codes, scale, q = make_data(g, n)
    a, b = make_tiles(codes, scale, q)
    c = torch.zeros(g, M_PAD, V, dtype=torch.float32, device=DEVICE)

    ok = gdn_lq_gemv_probe(a, b, c)
    if not ok:
        print(f"G={g:5d} N={n}: op unavailable")
        return None
    torch.npu.synchronize()
    o = c[:, :n, :]
    err = rel_l2(o, reference(codes, scale, q))

    gdn_lq_gemv_probe(a, b, c)  # warm: executor cached
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        gdn_lq_gemv_probe(a, b, c)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return err, ms


print(f"{'G':>6} {'N':>3} {'rel_L2':>12} {'ms':>9}")
results = {}
for g in (512, 2048, 8192):
    for n in (1, 2, 8):
        r = run(g, n)
        if r is None:
            sys.exit(1)
        err, ms = r
        results[(g, n)] = (err, ms)
        flag = "PASS" if err <= 3e-3 else "FAIL"
        print(f"{g:6d} {n:3d} {err:12.6f} {ms:9.3f}  {flag}")

bad = [(k, v[0]) for k, v in results.items() if v[0] > 3e-3]
if bad:
    print(f"PARITY FAIL: {bad}")
    sys.exit(1)
print("PARITY PASS (all N=1/2/8, all G)")
