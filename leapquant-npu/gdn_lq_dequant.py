"""M2-D2 stage 1: gdn_lq_dequant (AIV) — ctypes binding.

Per (slot, value-head): staging = fp16(codes) [T,h,V,K]; aq rows 0/1 =
fp16(kn/h//rep * s_k/127) and fp16(qn/h//rep * scale * s_k/127), rows 2:15
zero. Mirrors gemv_probe.py (vendor path registration + executor cache).
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
_AVAILABLE = None
_CACHE = {}
_CACHE_CAP = 256
_WS_BUF = None

# E1 localization experiment: executor/ctx lifecycle. C (default) is the
# historical behavior - cache the executor, destroy the CallContext.
# B keeps the context alive inside the cache (flush v2's fprep pattern).
# A never caches (fresh executor per call). Selected by
# VLLM_ASCEND_GDN_LQ_STEP_LIFECYCLE.
# Default B: the baked executor references the ctypes attr scalars
# (not copies); once they are GC'd every cached launch reads garbage -
# dequant/epilogue's slot_stride attr corruption is the E1 root cause
# (A/B/C experiment, 2026-10-07). C (destroy ctx, drop attrs) is kept
# for reproducing the corruption.
_LC = os.getenv("VLLM_ASCEND_GDN_LQ_STEP_LIFECYCLE", "D")


def _binding():
    global _B, _AVAILABLE
    if _AVAILABLE is False:
        return None
    if _B is None:
        cands = (glob.glob("/tmp/probe_opp_v*/vendors/custom_transformer/"
                           "op_api/lib/libcust_opapi.so")
                 + glob.glob("/usr/local/Ascend/opp/vendors/*/op_api/lib/"
                             "libcust_opapi.so", recursive=True)
                 + glob.glob("/mnt/workspace/vllm-ascend/vllm_ascend/"
                             "_cann_ops_custom/**/op_api/lib/libcust_opapi.so",
                             recursive=True))
        if not cands:
            _AVAILABLE = False
            return None
        try:
            lib = ctypes.CDLL(cands[0])
        except OSError:
            _AVAILABLE = False
            return None
        import torch_npu  # noqa: F401

        from fla_npu.ops.ascendc import _runtime as rt

        _B = (rt, rt.runtime(), lib)
        lib.aclnnGdnLqDequantGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnLqDequant.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.aclnnGdnLqDequant.restype = ctypes.c_int
    return _B


def _hc_loggate(n):
    # log hit #0 and every 200th afterwards
    return n == 0 or (n % 200) == 0

def launch(state, idx, kn, qn, staging, aq, num_heads, num_k_heads, scale,
           stream) -> bool:
    """cached fast path with a caller-provided stream; the caller performs
    the stream-capture check once for the whole fused step. Cache misses
    fall back to the full wrapper (slow path, once per key)."""
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    key = (int(state.data_ptr()), int(idx.data_ptr()),
           int(kn.data_ptr()), int(qn.data_ptr()),
           int(staging.data_ptr()), int(aq.data_ptr()),
           int(state.stride(0)), int(idx.shape[0]), int(num_heads),
           int(num_k_heads), float(scale))
    hit = _CACHE.get(key)
    global _WS_BUF
    if hit is not None and os.getenv("VLLM_ASCEND_GDN_LQ_STEPlog"):
        _hitc = getattr(launch, "_hits", 0)
        if _hc_loggate(_hitc):
            import sys as _s
            print(f"[stepv2-launch-hit] dequant n={_hitc} "
                  f"exec={hit[0]}", file=_s.stderr, flush=True)
        launch._hits = _hitc + 1
    if hit is None:
        return gdn_lq_dequant(state, idx, kn, qn, staging, aq, num_heads,
                              num_k_heads, scale)
    executor = hit[0]
    ret = lib.aclnnGdnLqDequant(
        ctypes.c_void_p(int(_WS_BUF.data_ptr())),
        ctypes.c_uint64(_WS_BUF.numel()),
        executor, ctypes.c_void_p(stream))
    return ret == 0


def gdn_lq_dequant(state, idx, kn, qn, staging, aq, num_heads, num_k_heads,
                   scale) -> bool:
    """stage 1 dequant. state: fp32 ssm_state slot buffer (strided ok);
    idx int64 [T]; kn/qn bf16 [T, nk, K] (key-head rows, unexpanded);
    staging fp16 out [T, h, V, K]; aq fp16 out [T, h, 16, K]."""
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except Exception:
        return False
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    dev = state.device

    # fast path: key from the raw tensors; dtype/contiguity conversion and
    # executor creation only happen on a miss (the conversions cost ~0.2 ms
    # of Python per call, comparable to the kernels themselves)
    stream = int(torch.npu.current_stream().npu_stream)
    key = (int(state.data_ptr()), int(idx.data_ptr()),
           int(kn.data_ptr()), int(qn.data_ptr()),
           int(staging.data_ptr()), int(aq.data_ptr()),
           int(state.stride(0)), int(idx.shape[0]), int(num_heads),
           int(num_k_heads), float(scale))
    hit = _CACHE.get(key)
    global _WS_BUF
    if _LC == "A":
        hit = None  # recreate the executor every call
    if os.getenv("VLLM_ASCEND_GDN_LQ_STEPlog") and hit is None:
        import sys as _s
        print(f"[stepv2-miss] dequant key={key}", file=_s.stderr,
              flush=True)
    if hit is not None:
        if os.getenv("VLLM_ASCEND_GDN_LQ_STEPlog"):
            import sys as _s
            print(f"[stepv2-hit] dequant exec={hit[0]} "
                  f"key0={key[0]:#x} idxptr={key[1]:#x}", file=_s.stderr,
                  flush=True)
        executor = hit[0]
        ret = lib.aclnnGdnLqDequant(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(_WS_BUF.numel()),
            executor, ctypes.c_void_p(stream))
        return ret == 0

    def _c(x, dt):
        x = x.to(device=dev, dtype=dt)
        return x if x.is_contiguous() else x.contiguous()

    try:
        # serialize executor creation against in-flight kernels: rapid
        # back-to-back _CallContext creation across two custom ops raced
        # with the previous kernel still on the stream and faulted vector
        # cores (observed 2026-10-05, see results/M2-D2-stepv2.md). The
        # slow path runs once per executor key, so a sync here is free.
        # state: pass a 1-element base view (as_strided works on any
        # layout, no copy). _c's .contiguous() would materialize the whole
        # strided pool view (GiB-scale) and the executor cache would keep it
        # alive forever - that was the decode-loop memory leak. The kernel
        # addresses slots through the base pointer + s_slot stride attr.
        slot_stride_bytes = state.stride(0) * state.element_size()
        state = torch.as_strided(state, (1,), (1,))
        idx = _c(idx, torch.int64)
        kn = _c(kn, torch.bfloat16)
        qn = _c(qn, torch.bfloat16)
        staging = _c(staging, torch.float16)
        aq = _c(aq, torch.float16)
        torch.npu.synchronize()
        ctx = rt._CallContext(runtime, dev)
    except Exception:
        return False
    try:
        # E1: keep the attr scalars alive for variant B - if the baked
        # executor references (not copies) them, their GC invalidates
        # every cached launch (dequant's slot_stride attr would corrupt
        # state addressing, matching the observed corruption)
        attrs = [ctypes.c_int64(int(num_heads)),
                 ctypes.c_int64(int(num_k_heads)),
                 ctypes.c_int64(int(slot_stride_bytes)),
                 ctypes.c_double(float(scale))]
        args = [
            ctx.tensor(state, "state"),
            ctx.tensor(idx, "idx"),
            ctx.tensor(kn, "kn"),
            ctx.tensor(qn, "qn"),
            *attrs,
            ctx.tensor(staging, "staging"),
            ctx.tensor(aq, "aq"),
        ]
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnLqDequantGetWorkspaceSize(
            *args, ctypes.byref(ws_size), ctypes.byref(executor))
        assert ret == 0, f"get_ws ret={ret} ws={ws_size.value}"
        # floor the shared workspace so it never reallocates: the cached
        # executor bakes the workspace ADDRESS at creation, and a later
        # realloc frees the old block for the caching allocator - the next
        # torch allocation lands on it and the stale executor corrupts it
        # (found via input-ring reuse at step 5, 2026-10-06)
        need = max(int(ws_size.value) if ws_size.value else 1, 256 * 2 ** 20)
        if _WS_BUF is None or _WS_BUF.numel() < need:
            _WS_BUF = torch.empty(need, dtype=torch.uint8, device=dev)
        ret = lib.aclnnGdnLqDequant(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(need), executor,
            ctypes.c_void_p(stream))
        assert ret == 0, f"launch ret={ret}"
        if _LC != "A":
            if len(_CACHE) >= _CACHE_CAP:
                _CACHE.clear()
            if _LC in ("B", "D"):
                # keep the attr scalars alive: the baked executor
                # references them (E1 root cause). B ALSO holds the
                # whole CallContext; D destroys it - B deadlocks in
                # long runs (ctx resource pinning, observed 2026-10-07
                # at every batch size), D is the production form.
                if _LC == "B":
                    _CACHE[key] = (executor, state, idx, kn, qn, staging, aq, ctx, attrs)
                else:
                    _CACHE[key] = (executor, state, idx, kn, qn, staging, aq, attrs)
            else:
                _CACHE[key] = (executor, state, idx, kn, qn, staging, aq)
        return True
    except Exception as e:
        import sys
        print(f"[gdn-lq-dequant] slow-path exception: "
              f"{type(e).__name__}: {e!r}", file=sys.stderr, flush=True)
        return False
    finally:
        if _LC != "B":
            ctx.destroy()


_WARMED = False
# executor bakes device addresses: the warm-up tensors MUST stay alive
# for the process lifetime or a reused address triggers stale-executor
# launches into unrelated live memory (random vec-core faults observed
# 2026-10-06).
_WARM_KEEP = None


def warm_cache(device="npu"):
    """One dummy slow-path call + sync, tensors kept alive forever."""
    global _WARMED, _WARM_KEEP
    if _WARMED:
        return True
    t, h, nk, v, k = 1, 2, 1, 128, 128
    keep = (
        torch.zeros(4, h, v, k, dtype=torch.float32, device=device),
        torch.ones(t, dtype=torch.int64, device=device),
        torch.zeros(t, nk, k, dtype=torch.bfloat16, device=device),
        torch.zeros(t, nk, k, dtype=torch.bfloat16, device=device),
        torch.zeros(t, h, v, k, dtype=torch.float16, device=device),
        torch.zeros(t, h, 16, k, dtype=torch.float16, device=device),
    )
    ok = gdn_lq_dequant(keep[0], keep[1], keep[2], keep[3], keep[4],
                        keep[5], h, nk, k ** -0.5)
    torch.npu.synchronize()
    _WARM_KEEP = keep
    _WARMED = ok
    return ok
