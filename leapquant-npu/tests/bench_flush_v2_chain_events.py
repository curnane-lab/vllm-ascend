"""In-chain device-event profile for flush v2 stages g1..t3.

``_chain7`` enqueues seven kernels on one stream. A device event is
recorded before each stage and after the last stage. fprep stays outside
this chain, matching the 2026-10-07 910C table.

The reported interval is event-to-event time on that stream. It includes
device idle if the host has not yet enqueued the kernel when the previous
event completes. On the published 910C run the short GEMMs at B=16 are
0.019 ms, so that idle is below the t3 interval.
"""

import ctypes
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
sys.path.insert(0, str(_REPO / "leapquant-npu"))

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
import gdn_lq_flush_v2 as v2  # noqa: E402

DEVICE = "npu"
H = 32
STAGE_NAMES = (
    "g1_S=AtB_K192",
    "g2_m1_Stq0",
    "t1_qk_mgs",
    "g3_qv_Sqk",
    "t2_qv_mgs",
    "g4_uk_Stvq",
    "t3_quantize",
)


def make(batch):
    generator = torch.Generator().manual_seed(11)
    idx = (torch.arange(batch) + 1).to(DEVICE)
    state = torch.randn(batch + 8, H, 128, 128, generator=generator).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(state)
    pools.init_from_dense(state, idx)
    pools.fill.zero_()
    pools.fill[:] = v2._WINDOW
    pools.w.zero_()
    return pools, idx


def emit(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def stage_launchers(pools, idx, batch):
    flags, ab, ba, s, c, kq, vq = v2._bufs(batch, H, idx.device)
    q0p = v2._q0_pad(pools)
    state_base = torch.as_strided(pools._state, (1,), (1,))
    slot_stride = pools._state.stride(0) * 4

    rt, runtime, lib = v2._binding()
    stream = int(torch.npu.current_stream().npu_stream)
    ctx = rt._CallContext(runtime, idx.device)
    keep = []
    descriptors = {
        "state": ctx.tensor(state_base, "state"),
        "idx": ctx.tensor(idx, "idx"),
        "s": ctx.tensor(s, "s_stage"),
        "ab": ctx.tensor(ab, "ab_stage"),
        "flags": ctx.tensor(flags, "flags"),
        "ba": ctx.tensor(ba, "ba_stage"),
        "q0": ctx.tensor(q0p, "q0_pad"),
        "kq": ctx.tensor(kq, "kq_stage"),
        "vq": ctx.tensor(vq, "vq_stage"),
        "c": ctx.tensor(c, "c_stage"),
    }
    gemv_pre = [descriptors[name] for name in
                ("state", "idx", "s", "ab", "flags", "ba", "q0", "kq", "vq")]
    tail_pre = [descriptors[name] for name in ("state", "idx", "s", "c")]
    chain = (
        ("GdnLqFgemv", gemv_pre, [descriptors["c"]], [H, slot_stride, 1]),
        ("GdnLqFgemv", gemv_pre, [descriptors["c"]], [H, slot_stride, 2]),
        ("GdnLqFtail", tail_pre, [descriptors["vq"], descriptors["kq"]],
         [H, slot_stride, 1]),
        ("GdnLqFgemv", gemv_pre, [descriptors["c"]], [H, slot_stride, 3]),
        ("GdnLqFtail", tail_pre, [descriptors["vq"], descriptors["kq"]],
         [H, slot_stride, 2]),
        ("GdnLqFgemv", gemv_pre, [descriptors["c"]], [H, slot_stride, 4]),
        ("GdnLqFtail", tail_pre, [descriptors["vq"], descriptors["kq"]],
         [H, slot_stride, 3]),
    )

    def stage_fn(index):
        name, pre, post, attrs = chain[index]

        def launch():
            scalars = [ctypes.c_int64(int(value)) for value in attrs]
            keep.append(scalars)
            workspace = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            get_ws = getattr(lib, f"aclnn{name}GetWorkspaceSize")
            run = getattr(lib, f"aclnn{name}")
            ret = get_ws(*pre, *scalars, *post,
                         ctypes.byref(workspace), ctypes.byref(executor))
            if ret != 0:
                return False
            need = int(workspace.value) if workspace.value else 1
            if v2._WS_BUF is None or v2._WS_BUF.numel() < need:
                v2._WS_BUF = torch.empty(need, dtype=torch.uint8, device=idx.device)
            return run(ctypes.c_void_p(int(v2._WS_BUF.data_ptr())),
                       ctypes.c_uint64(v2._WS_BUF.numel()),
                       executor,
                       ctypes.c_void_p(stream)) == 0

        return launch

    return [(STAGE_NAMES[index], stage_fn(index)) for index in range(len(chain))], ctx


def profile(batch, iterations, output):
    pools, idx = make(batch)
    stages, context = stage_launchers(pools, idx, batch)
    try:
        for _, launch in stages:
            assert launch(), "executor creation failed; audit availability"
        torch.npu.synchronize()

        stage_events = [0.0 for _ in stages]
        chain_wall = 0.0
        for _ in range(iterations):
            torch.npu.synchronize()
            events = [torch.npu.Event(enable_timing=True) for _ in range(len(stages) + 1)]
            chain_start = time.perf_counter()
            events[0].record()
            for stage_index, (_, launch) in enumerate(stages):
                assert launch()
                events[stage_index + 1].record()
            torch.npu.synchronize()
            chain_wall += time.perf_counter() - chain_start
            for stage_index in range(len(stages)):
                stage_events[stage_index] += events[stage_index].elapsed_time(
                    events[stage_index + 1])

        rows = []
        for stage_index, (name, _) in enumerate(stages):
            rows.append({
                "batch": batch,
                "stage": name,
                "device_interval_ms": round(stage_events[stage_index] / iterations, 3),
            })
        rows.append({
            "batch": batch,
            "stage": "chain_total",
            "chain_wall_ms": round(chain_wall / iterations * 1e3, 3),
        })
        for row in rows:
            emit(output, row)
    finally:
        context.destroy()


if __name__ == "__main__":
    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")
    for batch_size in (16, 64, 256):
        profile(batch_size, 30, output_path)
