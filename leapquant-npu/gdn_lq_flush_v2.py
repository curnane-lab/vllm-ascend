"""M3-3b flush v2: three-op cube pipeline orchestrator + ctypes bindings.

Chain per flush call (stream-ordered, due = fill >= L scanned in-kernel):
  fprep(AIV) -> fgemv(1: S=A^T B fp16) -> fgemv(2: m1=S^T q0)
  -> ftail(1: qk=mgs(m1)) -> fgemv(3: qv=S qk)
  -> ftail(2: qv'=mgs(qv)) -> fgemv(4: uk=S^T qv')
  -> ftail(3: E/quantize/writeback)
gdn_lq_flush (v1) remains the fallback path.
"""

import ctypes
import glob
import os

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


def _slow_path(lib, rt, runtime, dev, name, tensor_args, attrs, tensors):
    """generic slow-path executor creation. tensor_args: list of
    (tensor, name) or ('attr_i64', value); attrs appended before outputs."""
    torch.npu.synchronize()
    ctx = rt._CallContext(runtime, dev)
    try:
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
        return (run, executor)
    finally:
        ctx.destroy()


def _launch(run_executor, stream):
    run, executor = run_executor
    ret = run(ctypes.c_void_p(int(_WS_BUF.data_ptr())),
              ctypes.c_uint64(_WS_BUF.numel()), executor,
              ctypes.c_void_p(stream))
    return ret == 0


_CACHE = {}


def _fprep(state_base, idx, ab, flags, ba, h, s_slot, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    stream = int(torch.npu.current_stream().npu_stream)
    hit = _CACHE.get(key)
    if hit is not None:
        return _launch(hit, stream)
    dev = idx.device
    run_ex = _slow_path(lib, rt, runtime, dev, "GdnLqFprep",
                        [(state_base, "state"), (idx, "idx")],
                        [h, s_slot],
                        [(state_base, "state"), (idx, "idx"),
                         (ab, "ab_stage"), (flags, "flags"),
                         (ba, "ba_stage")])
    ok = _launch(run_ex, stream)
    if ok:
        _CACHE[key] = run_ex
    return ok


def _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c, h, s_slot,
           phase, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    stream = int(torch.npu.current_stream().npu_stream)
    hit = _CACHE.get(key)
    if hit is not None:
        return _launch(hit, stream)
    dev = idx.device
    run_ex = _slow_path(
        lib, rt, runtime, dev, "GdnLqFgemv",
        [(state_base, "state"), (idx, "idx"), (s, "s_stage"),
         (ab, "ab_stage"), (flags, "flags"), (ba, "ba_stage"),
         (q0p, "q0_pad"), (kq, "kq_stage"), (vq, "vq_stage")],
        [h, s_slot, phase],
        [(state_base, "state"), (idx, "idx"), (s, "s_stage"),
         (ab, "ab_stage"), (flags, "flags"), (ba, "ba_stage"),
         (q0p, "q0_pad"), (kq, "kq_stage"), (vq, "vq_stage"),
         (c, "c_stage")])
    ok = _launch(run_ex, stream)
    if ok:
        _CACHE[key] = run_ex
    return ok


def _ftail(state_base, idx, s, c, vq, kq, h, s_slot, phase, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    stream = int(torch.npu.current_stream().npu_stream)
    hit = _CACHE.get(key)
    if hit is not None:
        return _launch(hit, stream)
    dev = idx.device
    run_ex = _slow_path(lib, rt, runtime, dev, "GdnLqFtail",
                        [(state_base, "state"), (idx, "idx"),
                         (s, "s_stage"), (c, "c_stage")],
                        [h, s_slot, phase],
                        [(state_base, "state"), (idx, "idx"),
                         (s, "s_stage"), (c, "c_stage"), (vq, "vq_stage"),
                         (kq, "kq_stage")])
    ok = _launch(run_ex, stream)
    if ok:
        _CACHE[key] = run_ex
    return ok


_BUF_CACHE = {}
_Q0_CACHE = {}


def _bufs(t, h, dev):
    key = (t, h, str(dev))
    hit = _BUF_CACHE.get(key)
    if hit is None:
        hit = (
            torch.zeros(t, h, 32, 128, dtype=torch.int32, device=dev),
            torch.zeros(t, h, 192, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 192, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 16, dtype=torch.float32, device=dev),
            torch.zeros(t, h, 16, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 16, 128, dtype=torch.float16, device=dev),
        )
        _BUF_CACHE[key] = hit
    return hit


def _q0_pad(pools):
    key = int(pools.q0.data_ptr())
    hit = _Q0_CACHE.get(key)
    if hit is None:
        q0p = torch.zeros(128, 16, dtype=torch.float16, device=pools.q0.device)
        q0p[:, :4] = pools.q0.half()
        _Q0_CACHE.clear()
        _Q0_CACHE[key] = q0p
        hit = q0p
    return hit


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
        flags, ab, ba, s, c, kq, vq = _bufs(t, h, dev)
        q0p = _q0_pad(pools)
        state_base = torch.as_strided(pools._state, (1,), (1,))
        slot_stride = pools._state.stride(0) * 4
        base = (int(idx.data_ptr()), int(pools._state.data_ptr()), t, h)
        ok = _fprep(state_base, idx, ab, flags, ba, h, slot_stride,
                    ("prep",) + base)
        if not ok:
            return False
        # host-side due gate: fprep wrote per-instance flags; skip the
        # remaining 7 launches when nothing is due (AIC scalar GM reads
        # are broken on this stack, so the cube stages cannot check due
        # in-kernel - the cost is one small reduce + sync here)
        if int(flags.sum().item()) == 0:
            return True
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
