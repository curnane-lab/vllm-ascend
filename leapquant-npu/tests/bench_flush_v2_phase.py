"""E3-T2 per-phase device profile of the flush v2 chain (forced all-due).

Times each stage (fprep, g1..g4, t1..t3) individually with a sync on
both sides, 30 iterations, B=16/64/256. Raw rows ->
runs/flush_v2_phase_profile.jsonl
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

DEVICE = "npu"
H, V, K = 32, 128, 128
OUT = "runs/flush_v2_phase_profile.jsonl"


def make(b, seed=11):
    gen = torch.Generator().manual_seed(seed)
    idx = (torch.arange(b) + 1).to(DEVICE)
    buf = torch.randn(b + 8, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    pools.fill[:] = v2._WINDOW
    pools.w.zero_()
    return pools, idx


def emit(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def stage_timers(pools, idx, b):
    flags, ab, ba, s, c, kq, vq = v2._bufs(b, H, idx.device)
    q0p = v2._q0_pad(pools)
    state_base = torch.as_strided(pools._state, (1,), (1,))
    ss = pools._state.stride(0) * 4
    bk = (int(idx.data_ptr()), int(pools._state.data_ptr()), b, H)
    rt, runtime, lib = v2._binding()
    dev = idx.device

    def mk(name, key, phase):
        def fn():
            if name == "prep":
                v2._fprep(state_base, idx, ab, flags, ba, H, ss, ("prep",) + bk)
            elif name == "gemv":
                v2._fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                          H, ss, phase, (f"g{phase}",) + bk)
            else:
                v2._ftail(state_base, idx, s, c, vq, kq, H, ss, phase,
                          (f"t{phase}",) + bk)
        return fn

    return [
        ("fprep", mk("prep", None, None)),
        ("g1_S=AtB_K192", mk("gemv", None, 1)),
        ("g2_m1_Stq0", mk("gemv", None, 2)),
        ("t1_qk_mgs", mk("tail", None, 1)),
        ("g3_qv_Sqk", mk("gemv", None, 3)),
        ("t2_qv_mgs", mk("tail", None, 2)),
        ("g4_uk_Stvq", mk("gemv", None, 4)),
        ("t3_quantize", mk("tail", None, 3)),
    ]


def main():
    iters = 30
    for b in (16, 64, 256):
        pools, idx = make(b)
        stages = stage_timers(pools, idx, b)
        for name, fn in stages:
            fn()
            torch.npu.synchronize()
            tot = 0.0
            for _ in range(iters):
                torch.npu.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.npu.synchronize()
                tot += time.perf_counter() - t0
            emit({"batch": b, "stage": name,
                  "ms": round(tot / iters * 1e3, 3), "iters": iters})


if __name__ == "__main__":
    main()
