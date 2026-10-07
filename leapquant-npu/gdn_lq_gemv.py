"""M2-D2 stage 2: gdn_lq_gemv (AIC) — ctypes binding.

cq[t,h,v,0/1] = sum_k staging[t,h,v,k] * aq[t,h,0/1,k] on the cube.
Mirrors gemv_probe.py (vendor path registration + executor cache).
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
        lib.aclnnGdnLqGemvGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnLqGemv.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.aclnnGdnLqGemv.restype = ctypes.c_int
    return _B


def launch(staging, aq, cq, stream) -> bool:
    """cached fast path with a caller-provided stream; the caller performs
    the stream-capture check once for the whole fused step. Cache misses
    fall back to the full wrapper (slow path, once per key)."""
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    key = (int(staging.data_ptr()), int(aq.data_ptr()), int(cq.data_ptr()),
           int(staging.shape[0]), int(staging.shape[1]))
    hit = _CACHE.get(key)
    global _WS_BUF
    if hit is None:
        return gdn_lq_gemv(staging, aq, cq)
    executor = hit[0]
    ret = lib.aclnnGdnLqGemv(
        ctypes.c_void_p(int(_WS_BUF.data_ptr())),
        ctypes.c_uint64(_WS_BUF.numel()),
        executor, ctypes.c_void_p(stream))
    return ret == 0


def gdn_lq_gemv(staging, aq, cq) -> bool:
    """staging fp16 [T,h,V,K], aq fp16 [T,h,16,K] -> cq fp32 [T,h,V,16]."""
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except Exception:
        return False
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    dev = staging.device

    # fast path: key from the raw tensors; dtype/contiguity conversion and
    # executor creation only happen on a miss (the conversions cost ~0.2 ms
    # of Python per call, comparable to the kernels themselves)
    stream = int(torch.npu.current_stream().npu_stream)
    key = (int(staging.data_ptr()), int(aq.data_ptr()), int(cq.data_ptr()),
           int(staging.shape[0]), int(staging.shape[1]))
    hit = _CACHE.get(key)
    global _WS_BUF
    if _LC == "A":
        hit = None  # recreate the executor every call
    if os.getenv("VLLM_ASCEND_GDN_LQ_STEPlog") and hit is None:
        import sys as _s
        print(f"[stepv2-miss] gemv key={key}", file=_s.stderr,
              flush=True)
    if hit is not None:
        if os.getenv("VLLM_ASCEND_GDN_LQ_STEPlog"):
            import sys as _s
            print(f"[stepv2-hit] gemv exec={hit[0]} "
                  f"key0={key[0]:#x} idxptr={key[1]:#x}", file=_s.stderr,
                  flush=True)
        executor = hit[0]
        ret = lib.aclnnGdnLqGemv(
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
        staging = _c(staging, torch.float16)
        aq = _c(aq, torch.float16)
        cq = _c(cq, torch.float32)
        torch.npu.synchronize()
        ctx = rt._CallContext(runtime, dev)
    except Exception:
        return False
    try:
        args = [
            ctx.tensor(staging, "staging"),
            ctx.tensor(aq, "aq"),
            ctx.tensor(cq, "cq"),
        ]
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnLqGemvGetWorkspaceSize(
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
        ret = lib.aclnnGdnLqGemv(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(need), executor,
            ctypes.c_void_p(stream))
        assert ret == 0, f"launch ret={ret}"
        if _LC != "A":
            if len(_CACHE) >= _CACHE_CAP:
                _CACHE.clear()
            if _LC == "B":
                # hold the CallContext: the executor references its
                # tensor descriptors; destroying them is the E1
                # corruption suspect
                _CACHE[key] = (executor, staging, aq, cq, ctx)
            else:
                _CACHE[key] = (executor, staging, aq, cq)
        return True
    except Exception as e:
        import sys
        print(f"[gdn-lq-gemv] slow-path exception: "
              f"{type(e).__name__}: {e!r}", file=sys.stderr, flush=True)
        return False
    finally:
        if _LC != "B":
            ctx.destroy()


_WARMED = False
# see gdn_lq_dequant._WARM_KEEP: warm-up tensors must outlive the executor
_WARM_KEEP = None


def warm_cache(device="npu"):
    """One dummy slow-path call + sync, tensors kept alive forever."""
    global _WARMED, _WARM_KEEP
    if _WARMED:
        return True
    t, h, v, k = 1, 2, 128, 128
    keep = (
        torch.zeros(t, h, v, k, dtype=torch.float16, device=device),
        torch.zeros(t, h, 16, k, dtype=torch.float16, device=device),
        torch.zeros(t, h, v, 16, dtype=torch.float32, device=device),
    )
    ok = gdn_lq_gemv(keep[0], keep[1], keep[2])
    torch.npu.synchronize()
    _WARM_KEEP = keep
    _WARMED = ok
    return ok
