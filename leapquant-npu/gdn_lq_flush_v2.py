"""M3-3b flush v2: three-op cube pipeline orchestrator + ctypes bindings.

Chain per flush call (stream-ordered, due = fill >= L scanned in-kernel):
  fprep(AIV) -> fgemv(1: S=A^T B fp16) -> fgemv(2: m1=S^T q0)
  -> ftail(1: qk=mgs(m1)) -> fgemv(3: qv=S qk)
  -> ftail(2: qv'=mgs(qv)) -> fgemv(4: uk=S^T qv')
  -> ftail(3: E/quantize/writeback)
gdn_lq_flush (v1) remains the fallback path.

Due gating without host sync (M3-3b fix): fprep writes each head's fill
VALUE into ``flags``; the wrapper copies flags to pinned memory
(asynchronously) and decides the remaining 7 launches from LAST step's
pinned values: a head is due this step iff its fill was L-1 last step
(decode advances fill by exactly one per step; resync resets it to 0).
Two consecutive fill >= L readings also trigger (flush called without an
intervening decode step - bench loops only; production always decodes,
wrapping fill to 1). Copy completion is proven by a device round-tripped
sequence number (torch.npu.Event.query() is unreliable on this stack),
never by synchronize. Steady-state nothing-due costs one fprep launch
plus two tiny async copies.
"""

import ctypes
import glob
import os

import numpy as np
import torch

_VENDOR = ("/mnt/workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/"
           "vendors/custom_transformer")
if os.path.isdir(_VENDOR):
    _cur = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    _paths = [p for p in _cur.split(":") if p]
    if _VENDOR not in _paths:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            _VENDOR + (":" + _cur if _cur else ""))

_B = None
_WS_BUF = None


def _binding():
    global _B
    if _B is None:
        cands = (glob.glob("/tmp/probe_opp_v*/vendors/custom_transformer/"
                           "op_api/lib/libcust_opapi.so")
                 + glob.glob("/usr/local/Ascend/opp/vendors/*/op_api/lib/"
                             "libcust_opapi.so", recursive=True)
                 + glob.glob("/mnt/workspace/vllm-ascend/vllm_ascend/"
                             "_cann_ops_custom/**/op_api/lib/libcust_opapi.so",
                             recursive=True))
        if not cands:
            raise RuntimeError("libcust_opapi.so not found")
        lib = ctypes.CDLL(cands[0])
        import torch_npu  # noqa: F401
        from fla_npu.ops.ascendc import _runtime as rt
        _B = (rt, rt.runtime(), lib)
    return _B


def _create(lib, rt, runtime, dev, name, tensor_args, attrs, tensors):
    """create an executor; the CallContext MUST stay alive while the
    executor is used (it owns the tensor descriptors)."""
    ctx = rt._CallContext(runtime, dev)
    args = []
    for t, nm in tensor_args:
        args.append(ctx.tensor(t, nm))
    for v in attrs:
        args.append(ctypes.c_int64(int(v)))
    for t, nm in tensors[len(tensor_args):]:
        args.append(ctx.tensor(t, nm))
    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()
    get_ws = getattr(lib, f"aclnn{name}GetWorkspaceSize")
    run = getattr(lib, f"aclnn{name}")
    ret = get_ws(*args, ctypes.byref(ws_size), ctypes.byref(executor))
    assert ret == 0, f"{name} get_ws ret={ret}"
    global _WS_BUF
    need = int(ws_size.value) if ws_size.value else 1
    if _WS_BUF is None or _WS_BUF.numel() < need:
        _WS_BUF = torch.empty(need, dtype=torch.uint8, device=dev)
    return (run, executor, ctx)


def _launch(run_executor, stream):
    run, executor, _ctx = run_executor
    ret = run(ctypes.c_void_p(int(_WS_BUF.data_ptr())),
              ctypes.c_uint64(_WS_BUF.numel()), executor,
              ctypes.c_void_p(stream))
    return ret == 0


def _slow_launch(lib, rt, runtime, dev, name, tensor_args, attrs, tensors):
    """create + launch + destroy, every call. REQUIRED for multi-phase
    ops (GdnLqFgemv/GdnLqFtail): the aclnn runtime pools executors by op
    type + tensor descriptors with the ATTRS EXCLUDED from the key, so
    the phases collide on ONE pool slot and the last-created config wins
    the shared handle - caching phase-specific executors silently runs
    the wrong phase's tiling (observed: a cached g1 returned zeros after
    g2's executor was created). run() snapshots the launch at enqueue
    (verified: an in-flight task is unaffected by the next executor
    creation), so recreating between stages is safe."""
    run_ex = _create(lib, rt, runtime, dev, name, tensor_args, attrs, tensors)
    stream = int(torch.npu.current_stream().npu_stream)
    ok = _launch(run_ex, stream)
    run_ex[2].destroy()
    return ok


_CACHE = {}


def _fprep(state_base, idx, ab, flags, ba, h, s_slot, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    stream = int(torch.npu.current_stream().npu_stream)
    hit = _CACHE.get(key)
    if hit is not None:
        return _launch(hit, stream)
    dev = idx.device
    run_ex = _create(lib, rt, runtime, dev, "GdnLqFprep",
                     [(state_base, "state"), (idx, "idx")],
                     [h, s_slot],
                     [(state_base, "state"), (idx, "idx"),
                      (ab, "ab_stage"), (flags, "flags"),
                      (ba, "ba_stage")])
    ok = _launch(run_ex, stream)
    if ok:
        # single-phase op: one executor per descriptor set, cacheable
        _CACHE[key] = run_ex
    return ok


def _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c, h, s_slot,
           phase, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    return _slow_launch(
        lib, rt, runtime, idx.device, "GdnLqFgemv",
        [(state_base, "state"), (idx, "idx"), (s, "s_stage"),
         (ab, "ab_stage"), (flags, "flags"), (ba, "ba_stage"),
         (q0p, "q0_pad"), (kq, "kq_stage"), (vq, "vq_stage")],
        [h, s_slot, phase],
        [(state_base, "state"), (idx, "idx"), (s, "s_stage"),
         (ab, "ab_stage"), (flags, "flags"), (ba, "ba_stage"),
         (q0p, "q0_pad"), (kq, "kq_stage"), (vq, "vq_stage"),
         (c, "c_stage")])


def _ftail(state_base, idx, s, c, vq, kq, h, s_slot, phase, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    return _slow_launch(lib, rt, runtime, idx.device, "GdnLqFtail",
                        [(state_base, "state"), (idx, "idx"),
                         (s, "s_stage"), (c, "c_stage")],
                        [h, s_slot, phase],
                        [(state_base, "state"), (idx, "idx"),
                         (s, "s_stage"), (c, "c_stage"), (vq, "vq_stage"),
                         (kq, "kq_stage")])


_BUF_CACHE = {}
_Q0_CACHE = {}


def _bufs(t, h, dev):
    key = (t, h, str(dev))
    hit = _BUF_CACHE.get(key)
    if hit is None:
        # per-instance c/kq/vq regions are [128,32]: fgemv phases 2-4 run
        # with N=32 (MatmulApiTiling rejects N=16) and ftail stores/reads
        # the full [.,32] staging row. The previous [.,16] allocation was
        # half the addressed size and the OOB stores faulted (ftail
        # phase-2 vq store MTE invalid-GM-address).
        hit = (
            torch.zeros(t, h, 32, 128, dtype=torch.int32, device=dev),
            torch.zeros(t, h, 192, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 192, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 32, dtype=torch.float32, device=dev),
            torch.zeros(t, h, 128, 32, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 32, dtype=torch.float16, device=dev),
        )
        _BUF_CACHE[key] = hit
    return hit


def _q0_pad(pools):
    key = int(pools.q0.data_ptr())
    hit = _Q0_CACHE.get(key)
    if hit is None:
        # [128,32]: fgemv phases 2-4 address B with the padded N=32
        # geometry; the extra columns are never consumed (ftail reads
        # cols 0..3) but the address range must be mapped.
        q0p = torch.zeros(128, 32, dtype=torch.float16, device=pools.q0.device)
        q0p[:, :4] = pools.q0.half()
        _Q0_CACHE.clear()
        _Q0_CACHE[key] = q0p
        hit = q0p
    return hit


_WINDOW = 16
_MIRROR = {}


def _mirror(dev, n):
    """Pinned double buffer + sequence-tag validity for the sync-free due
    gate. torch.npu.Event is unusable on this stack (query() never fires
    and repeated record corrupted later kernels), so copy completion is
    proven by a device round-trip: the host bumps a 1-element device
    tensor before fprep and copies it back after; a buffer's pinned
    fills are trustworthy only when its companion pinned slot shows the
    matching sequence number."""
    key = str(dev)
    st = _MIRROR.get(key)
    if st is None or st["cap"] < n:
        cap = max(n, 4096)
        st = {
            "cap": cap,
            "pin": [
                torch.zeros(cap, dtype=torch.int32, pin_memory=True),
                torch.zeros(cap, dtype=torch.int32, pin_memory=True),
            ],
            "pin_seq": [
                torch.zeros(1, dtype=torch.int32, pin_memory=True),
                torch.zeros(1, dtype=torch.int32, pin_memory=True),
            ],
            "dev_seq": torch.zeros(1, dtype=torch.int32, device=dev),
            "host_seq": torch.zeros(1, dtype=torch.int32, pin_memory=True),
            # numpy views for the microsecond-scale skip decision
            "pin_np": None,
            "seq_np": None,
            # per-slot: gen of the last fills copy written there, the
            # batch size of that copy, and the steps simulated since
            "buf_gen": [0, 0],
            "buf_n": [0, 0],
            "age": 0,
            "gen": 0,
        }
        st["pin_np"] = [t.numpy() for t in st["pin"]]
        st["seq_np"] = [t.numpy() for t in st["pin_seq"]]
        _MIRROR[key] = st
    return st


def reset_due_gate(dev=None):
    """Drop the mirror (next flush refreshes it). For tests that advance
    the window outside the one-decode-per-flush production pattern."""
    if dev is None:
        _MIRROR.clear()
    else:
        _MIRROR.pop(str(dev), None)


def flush_pools_v2(pools, idx) -> bool:
    """Cube-pipeline flush (M3-3b). Returns False when unavailable."""
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except Exception:
        return False
    try:
        h = pools.num_heads
        t = idx.shape[0]
        dev = idx.device
        n = t * h
        # Sync-free due gate (production pattern: one decode step between
        # flushes): fill advances by exactly one per decode step, wraps
        # to 1 after a fold (fill == L), and resync resets it to 0, so
        # the host simulates fill from the freshest PROVEN pinned values
        # (a buffer is proven when its round-tripped seq number shows
        # up) and launches only when the simulated fill reaches L. Slot
        # churn at a position can only over-predict (launch is always
        # safe: ftail re-checks due on device); an under-predict would
        # need a fresh slot to be due earlier than the predecessor's
        # leftover fill allows, which the arithmetic rules out. Batch
        # shape change forces a refresh launch. Nothing-due steps cost
        # zero device work and no synchronize.
        st = _mirror(dev, n)
        best = -1
        bw = 0
        for w in (0, 1):
            g = st["buf_gen"][w]
            if g > 0 and int(st["seq_np"][w][0]) == g:
                if g > best:
                    best = g
                    bw = w
        if best > 0 and st["buf_n"][bw] == n:
            prev = st["pin_np"][bw][:n]
            base = np.where(prev >= _WINDOW, 0, prev)
            if not bool((base + st["age"] + 1 >= _WINDOW).any()):
                st["age"] += 1
                return True
        flags, ab, ba, s, c, kq, vq = _bufs(t, h, dev)
        q0p = _q0_pad(pools)
        state_base = torch.as_strided(pools._state, (1,), (1,))
        slot_stride = pools._state.stride(0) * 4
        base = (int(idx.data_ptr()), int(pools._state.data_ptr()), t, h)
        counter = st["gen"] + 1
        st["host_seq"][0] = counter
        # NB: Tensor.to(dst, non_blocking=True) is a SILENT NO-OP on this
        # torch_npu build - dst.copy_(src, non_blocking=True) is the
        # working async form (verified H2D and D2H).
        st["dev_seq"].copy_(st["host_seq"], non_blocking=True)
        ok = _fprep(state_base, idx, ab, flags, ba, h, slot_stride,
                    ("prep",) + base)
        if not ok:
            return False
        # Refresh the pinned mirror: alternate the write slot so the
        # previous launch's copy stays readable until this one proves.
        w = counter % 2
        st["pin"][w][:n].copy_(flags.view(torch.int32).reshape(-1)[:n],
                               non_blocking=True)
        st["pin_seq"][w].copy_(st["dev_seq"], non_blocking=True)
        st["buf_gen"][w] = counter
        st["buf_n"][w] = n
        st["gen"] = counter
        st["age"] = 0
        ok = _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                    h, slot_stride, 1, ("g1",) + base)
        ok = ok and _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                           h, slot_stride, 2, ("g2",) + base)
        ok = ok and _ftail(state_base, idx, s, c, vq, kq, h, slot_stride, 1,
                           ("t1",) + base)
        ok = ok and _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                           h, slot_stride, 3, ("g3",) + base)
        ok = ok and _ftail(state_base, idx, s, c, vq, kq, h, slot_stride, 2,
                           ("t2",) + base)
        ok = ok and _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                           h, slot_stride, 4, ("g4",) + base)
        ok = ok and _ftail(state_base, idx, s, c, vq, kq, h, slot_stride, 3,
                           ("t3",) + base)
        return ok
    except Exception as e:
        import sys
        print(f"[flush-v2] exception: {type(e).__name__}: {e!r}",
              file=sys.stderr, flush=True)
        return False
