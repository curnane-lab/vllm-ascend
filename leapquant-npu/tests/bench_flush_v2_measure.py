"""E5 trustworthy flush-v2 measurements, three separated scenarios.

Replaces the retired in-file benches (mirror-polluted: back-to-back
flushes without decode and fill=16 -> base=0 host skips leaked into the
"all-due" average, making 26.9 ms non-reproducible).

Scenarios (raw jsonl rows appended to runs/flush_v2_measure.jsonl):
 1. forced all-due   - VLLM_ASCEND_GDN_LQ_FLUSH_V2_FORCE=1, fill held at
                       L, one sync'd flush per iteration: the pure
                       8-stage chain cost, mirror fully bypassed
 2. window rhythm    - 64 decode+flush steps (4 windows), sync per step,
                       launch count accounted: the real per-step
                       amortization of the production pattern
 3. nothing-due      - clean mirror, fill=1 steady state, per-step host
                       wall of the skip path (median)

Usage: python tests/bench_flush_v2_measure.py [--scenario N]
"""

import json
import os
import sys
from pathlib import Path
import time

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
sys.path.insert(0, str(_REPO / "leapquant-npu"))

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
import gdn_lq_flush_v2 as v2  # noqa: E402
from gdn_lq_flush_v2 import (  # noqa: E402
    flush_pools_v2_strict,
)

DEVICE = "npu"
H, V, K = 32, 128, 128
OUT = "runs/flush_v2_measure.jsonl"


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def make(b, seed=11):
    gen = torch.Generator().manual_seed(seed)
    idx = (torch.arange(b) + 1).to(DEVICE)
    buf = torch.randn(b + 8, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    g = torch.Generator().manual_seed(seed + 1)
    nk = H // 2
    kn = _norm(torch.randn(b, nk, K, generator=g)).to(DEVICE)
    qn = _norm(torch.randn(b, nk, K, generator=g)).to(DEVICE)
    kn = kn.repeat_interleave(H // nk, dim=1)
    qn = qn.repeat_interleave(H // nk, dim=1)
    vx = torch.randn(b, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    gl = -(torch.rand(b, H, generator=g) * 0.1 + 0.005).to(DEVICE)
    bt = torch.rand(b, H, generator=g).to(torch.bfloat16).to(DEVICE)
    return pools, idx, (kn, qn, vx, gl, bt)


def emit(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def forced_all_due(b, iters=30):
    os.environ["VLLM_ASCEND_GDN_LQ_FLUSH_V2_FORCE"] = "1"
    try:
        pools, idx, _ = make(b)
        # hold every head due: fill = L
        pools.fill.zero_()
        pools.fill[:] = v2._WINDOW
        pools.w.zero_()
        for _ in range(3):  # warm the 8 executors
            flush_pools_v2_strict(pools, idx, expect_chain=True)
        torch.npu.synchronize()
        tot = 0.0
        for _ in range(iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            st = flush_pools_v2_strict(pools, idx, expect_chain=True)
            torch.npu.synchronize()
            tot += time.perf_counter() - t0
        ms = tot / iters * 1e3
        emit({"scenario": "forced_all_due", "batch": b, "ms": round(ms, 3),
              "iters": iters, "v2_stats": st})
    finally:
        del os.environ["VLLM_ASCEND_GDN_LQ_FLUSH_V2_FORCE"]


def window_rhythm(b, windows=4):
    pools, idx, ins = make(b)
    v2.reset_due_gate()
    pools.fill.zero_()
    pools.w.zero_()
    launches = 0
    torch.npu.synchronize()
    t_start = time.perf_counter()
    for s in range(1, 16 * windows + 1):
        pools.decode_step(idx, *ins, 1.0)
        st = v2._mirror(pools, idx.device, b * H)
        g0 = st["gen"]
        flush_pools_v2_strict(pools, idx, step=s)
        torch.npu.synchronize()
        if st["buf_launch"][st["gen"] % 2] and st["buf_gen"][st["gen"] % 2] == st["gen"]:
            launches += 1
    tot = (time.perf_counter() - t_start) * 1e3
    steps = 16 * windows
    emit({"scenario": "window_rhythm", "batch": b, "steps": steps,
          "launches": launches, "total_ms": round(tot, 1),
          "amortized_ms_per_step": round(tot / steps, 4)})


def nothing_due(b, iters=48):
    pools, idx, ins = make(b)
    v2.reset_due_gate()
    pools.fill.zero_()
    pools.w.zero_()
    pools.decode_step(idx, *ins, 1.0)   # fill=1
    flush_pools_v2_strict(pools, idx, step=1)  # cold launch
    walls = []
    for s in range(2, iters + 2):
        pools.decode_step(idx, *ins, 1.0)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        flush_pools_v2_strict(pools, idx, step=s)
        walls.append(time.perf_counter() - t0)
        torch.npu.synchronize()
    walls = sorted(walls)
    med = walls[len(walls) // 2] * 1e3
    emit({"scenario": "nothing_due", "batch": b,
          "median_host_ms": round(med, 4),
          "max_host_ms": round(walls[-1] * 1e3, 4)})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for b in (16, 64, 256):
        if which in ("all", "1"):
            forced_all_due(b)
        if which in ("all", "2"):
            window_rhythm(b)
        if which in ("all", "3"):
            nothing_due(b)
