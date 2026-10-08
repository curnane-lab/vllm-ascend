"""M3-3b ftail phase-2 minimal repro (fixed staging inputs).

The original driver allocated c/kq/vq staging as [t,h,128,16] / [t,h,16,128]
(2048 elems/instance) while the kernels address per-instance regions of
[128,32] (4096 elems, fgemv N=32 padded geometry): every instance past the
first half wrote past the tensor. On this allocation layout the ftail
phase-2 vq store deterministically hits an unmapped page (MTE
invalid-GM-address), while the identical phase-1 kq store lands on a
mapped neighbour - which is why the failure looked device-flaky.

Usage:
  python tests/debug_ftail_repro.py old   # expect MTE fault at t2
  python tests/debug_ftail_repro.py new   # expect chain completes + parity

Runs ONE mode per process: the old mode faults the device context.
"""

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
H, V, K, SLOTS = 32, 128, 128, 64
T = 16 if len(sys.argv) > 2 and sys.argv[2] == "big" else 4


def _norm(x):
    return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "new"
    cols = 16 if mode == "old" else 32
    torch.npu.set_device(0)

    gen = torch.Generator().manual_seed(5)
    idx = torch.arange(1, T + 1, dtype=torch.int64).to(DEVICE)
    buf = torch.randn(SLOTS, H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)

    g = torch.Generator().manual_seed(9)
    kn = _norm(torch.randn(T, H, K, generator=g)).to(DEVICE)
    qn = _norm(torch.randn(T, H, K, generator=g)).to(DEVICE)
    vx = torch.randn(T, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    gl = (torch.randn(T, H, generator=g) * 0.05).to(DEVICE)
    bt = torch.rand(T, H, generator=g).to(torch.bfloat16).to(DEVICE)
    for _ in range(16):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    torch.npu.synchronize()

    # reference: torch flush on a CPU shadow of the same trajectory
    ref = GDNLeapQuantPools(pools._state.to("cpu").clone())
    ref.flush(idx.cpu())

    flags = torch.zeros(T, H, 32, 128, dtype=torch.int32, device=DEVICE)
    ab = torch.zeros(T, H, 192, 128, dtype=torch.float16, device=DEVICE)
    ba = torch.zeros(T, H, 192, 128, dtype=torch.float16, device=DEVICE)
    s = torch.zeros(T, H, 128, 128, dtype=torch.float16, device=DEVICE)
    c = torch.zeros(T, H, 128, cols, dtype=torch.float32, device=DEVICE)
    kq = torch.zeros(T, H, 128, cols, dtype=torch.float16, device=DEVICE)
    vq = torch.zeros(T, H, 128, cols, dtype=torch.float16, device=DEVICE)
    q0p = torch.zeros(128, max(cols, 32), dtype=torch.float16, device=DEVICE)
    q0p[:, :4] = pools.q0.half()
    state_base = torch.as_strided(pools._state, (1,), (1,))
    slot_stride = pools._state.stride(0) * 4
    h = H
    base = (int(idx.data_ptr()), int(pools._state.data_ptr()), T, h)

    def stage(name, fn):
        t0 = time.perf_counter()
        ok = fn()
        torch.npu.synchronize()
        print(f"[repro] {name}: ok={ok} {1e3*(time.perf_counter()-t0):.3f} ms",
              flush=True)
        return ok

    ok = stage("fprep", lambda: v2._fprep(state_base, idx, ab, flags, ba,
                                          h, slot_stride, ("prep",) + base))
    ok = ok and stage("g1", lambda: v2._fgemv(
        state_base, idx, s, ab, flags, ba, q0p, kq, vq, c, h, slot_stride,
        1, ("g1",) + base))
    ok = ok and stage("g2", lambda: v2._fgemv(
        state_base, idx, s, ab, flags, ba, q0p, kq, vq, c, h, slot_stride,
        2, ("g2",) + base))
    ok = ok and stage("t1", lambda: v2._ftail(
        state_base, idx, s, c, vq, kq, h, slot_stride, 1, ("t1",) + base))
    ok = ok and stage("g3", lambda: v2._fgemv(
        state_base, idx, s, ab, flags, ba, q0p, kq, vq, c, h, slot_stride,
        3, ("g3",) + base))
    ok = ok and stage("t2", lambda: v2._ftail(
        state_base, idx, s, c, vq, kq, h, slot_stride, 2, ("t2",) + base))
    ok = ok and stage("g4", lambda: v2._fgemv(
        state_base, idx, s, ab, flags, ba, q0p, kq, vq, c, h, slot_stride,
        4, ("g4",) + base))
    ok = ok and stage("t3", lambda: v2._ftail(
        state_base, idx, s, c, vq, kq, h, slot_stride, 3, ("t3",) + base))
    if not ok:
        print("[repro] chain returned False", flush=True)
        sys.exit(2)

    shadow = GDNLeapQuantPools(pools._state.cpu())
    t_idx, h_idx = shadow.th_index(idx.cpu())
    sgot = shadow.reconstruct_dense(idx.cpu())
    sref = ref.reconstruct_dense(idx.cpu())
    rel = ((sgot - sref).norm() / sref.norm().clamp_min(1e-9)).item()
    fill = shadow.fill[t_idx, h_idx]
    print(f"[repro] mode={mode} decoded-state rel err vs torch flush: {rel:.6f}",
          flush=True)
    print(f"[repro] fill[0,0]={fill[0,0].item()} (expect {v2._WINDOW})",
          flush=True)
    ok = rel <= 0.02 and fill[0, 0].item() == v2._WINDOW
    print(f"[repro] {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
