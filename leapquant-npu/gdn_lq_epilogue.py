"""M2-D2 stage 3: gdn_lq_epilogue (AIV) — ctypes binding.

v1 GdnLqStep assemble half with codes dots from the stage-2 cube output.
Mirrors gdn_lq_dequant.py (vendor path + executor cache + warm keep-alive).
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

# E1 localization experiment: executor/ctx lifecycle. C is the historical
# behavior - cache the executor, destroy the CallContext and drop attrs.
# B keeps the context alive inside the cache (flush v2's fprep pattern).
# A never caches (fresh executor per call). Selected by
# VLLM_ASCEND_GDN_LQ_STEP_LIFECYCLE (read once at import).
# Default D: destroy ctx but keep the ctypes attr scalars alive in the
# cache tuple - the baked executor references them (not copies); once they
# are GC'd every cached launch reads garbage (A/B/C/D experiment,
# 2026-10-07). NOTE: until the epilogue attrs fix in the same round, the
# epilogue attrs list was never passed to GetWorkspaceSize at all, so its
# slow path always failed and variant D's smoke result was ambiguous
# (silent triton fallback is also 3/4 @36). Any retest must confirm the
# v2 path actually runs: check for absence of "[gdn-lq-epilogue]
# slow-path exception" and cache hits under VLLM_ASCEND_GDN_LQ_STEPlog=1.
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
        lib.aclnnGdnLqEpilogueGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnLqEpilogue.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.aclnnGdnLqEpilogue.restype = ctypes.c_int
    return _B


def launch(state, idx, cq, kn, qn, v, g_log, beta, o, num_heads,
           num_k_heads, scale, stream) -> bool:
    """cached fast path with a caller-provided stream; the caller performs
    the stream-capture check once for the whole fused step. Cache misses
    fall back to the full wrapper (slow path, once per key)."""
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    key = (int(state.data_ptr()), int(idx.data_ptr()),
           int(cq.data_ptr()),
           int(kn.data_ptr()), int(qn.data_ptr()), int(v.data_ptr()),
           int(g_log.data_ptr()), int(beta.data_ptr()), int(o.data_ptr()),
           int(state.stride(0)), int(idx.shape[0]), int(num_heads),
           int(num_k_heads), float(scale))
    hit = _CACHE.get(key)
    global _WS_BUF
    if hit is None:
        return gdn_lq_epilogue(state, idx, cq, kn, qn, v, g_log, beta, o,
                               num_heads, num_k_heads, scale)
    executor = hit[0]
    ret = lib.aclnnGdnLqEpilogue(
        ctypes.c_void_p(int(_WS_BUF.data_ptr())),
        ctypes.c_uint64(_WS_BUF.numel()),
        executor, ctypes.c_void_p(stream))
    return ret == 0


def gdn_lq_epilogue(state, idx, cq, kn, qn, v, g_log, beta, o, num_heads,
                    num_k_heads, scale) -> bool:
    """state fp32 slot buffer; idx int64 [T]; cq fp32 [T,h,16,V]; kn/qn bf16
    [T,nk,K] unexpanded; v bf16 [T,h,V]; g_log fp32 [T,h]; beta bf16 [T,h];
    o bf16 out [T,h,V]."""
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
           int(cq.data_ptr()),
           int(kn.data_ptr()), int(qn.data_ptr()), int(v.data_ptr()),
           int(g_log.data_ptr()), int(beta.data_ptr()), int(o.data_ptr()),
           int(state.stride(0)), int(idx.shape[0]), int(num_heads),
           int(num_k_heads), float(scale))
    hit = _CACHE.get(key)
    global _WS_BUF
    if _LC == "A":
        hit = None  # recreate the executor every call
    if os.getenv("VLLM_ASCEND_GDN_LQ_STEPlog") and hit is None:
        import sys as _s
        print(f"[stepv2-miss] epilogue key={key}", file=_s.stderr,
              flush=True)
    if hit is not None:
        if os.getenv("VLLM_ASCEND_GDN_LQ_STEPlog"):
            import sys as _s
            print(f"[stepv2-hit] epilogue exec={hit[0]} "
                  f"key0={key[0]:#x} idxptr={key[1]:#x}", file=_s.stderr,
                  flush=True)
        executor = hit[0]
        ret = lib.aclnnGdnLqEpilogue(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(_WS_BUF.numel()),
            executor, ctypes.c_void_p(stream))
        return ret == 0

    def _c(x, dt):
        x = x.to(device=dev, dtype=dt)
        return x if x.is_contiguous() else x.contiguous()

    try:
        # serialize executor creation against in-flight kernels (see
        # results/M2-D2-stepv2.md); slow path only, once per key
        # state: pass a 1-element base view (as_strided works on any
        # layout, no copy). _c's .contiguous() would materialize the whole
        # strided pool view (GiB-scale) and the executor cache would keep it
        # alive forever - that was the decode-loop memory leak. The kernel
        # addresses slots through the base pointer + s_slot stride attr.
        slot_stride_bytes = state.stride(0) * state.element_size()
        state = torch.as_strided(state, (1,), (1,))
        idx = _c(idx, torch.int64)
        cq = _c(cq, torch.float32)
        kn = _c(kn, torch.bfloat16)
        qn = _c(qn, torch.bfloat16)
        v = _c(v, torch.bfloat16)
        g_log = _c(g_log, torch.float32)
        beta = _c(beta, torch.bfloat16)
        o = _c(o, torch.bfloat16)
        torch.npu.synchronize()
        ctx = rt._CallContext(runtime, dev)
    except Exception:
        return False
    try:
        attrs = [
            ctypes.c_int64(int(num_heads)),
            ctypes.c_int64(int(num_k_heads)),
            ctypes.c_int64(int(slot_stride_bytes)),
            ctypes.c_double(float(scale)),
        ]
        args = [
            ctx.tensor(state, "state"),
            ctx.tensor(idx, "idx"),
            ctx.tensor(cq, "cq"),
            ctx.tensor(kn, "kn"),
            ctx.tensor(qn, "qn"),
            ctx.tensor(v, "v"),
            ctx.tensor(g_log, "g_log"),
            ctx.tensor(beta, "beta"),
            *attrs,
            ctx.tensor(o, "o"),
        ]
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnLqEpilogueGetWorkspaceSize(
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
        ret = lib.aclnnGdnLqEpilogue(
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
                    _CACHE[key] = (executor, state, idx, cq, kn, qn, v, g_log, beta, o, ctx, attrs)
                else:
                    _CACHE[key] = (executor, state, idx, cq, kn, qn, v, g_log, beta, o, attrs)
            else:
                _CACHE[key] = (executor, state, idx, cq, kn, qn, v, g_log, beta, o)
        return True
    except Exception as e:
        import sys
        print(f"[gdn-lq-epilogue] slow-path exception: "
              f"{type(e).__name__}: {e!r}", file=sys.stderr, flush=True)
        return False
    finally:
        if _LC != "B":
            ctx.destroy()


_WARMED = False
# executor bakes device addresses: warm-up tensors must outlive the executor
_WARM_KEEP = None


def warm_cache(device="npu"):
    """One dummy slow-path call + sync, tensors kept alive forever."""
    global _WARMED, _WARM_KEEP
    if _WARMED:
        return True
    t, h, nk, v, k = 1, 2, 1, 128, 128
    state = torch.zeros(4, h, v, k, dtype=torch.float32, device=device)
    idx = torch.ones(t, dtype=torch.int64, device=device)
    cq = torch.zeros(t, h, 16, v, dtype=torch.float32, device=device)
    kn = torch.zeros(t, nk, k, dtype=torch.bfloat16, device=device)
    qn = torch.zeros(t, nk, k, dtype=torch.bfloat16, device=device)
    vv = torch.zeros(t, h, v, dtype=torch.bfloat16, device=device)
    g = torch.zeros(t, h, dtype=torch.float32, device=device)
    beta = torch.zeros(t, h, dtype=torch.bfloat16, device=device)
    o = torch.zeros(t, h, v, dtype=torch.bfloat16, device=device)
    ok = gdn_lq_epilogue(state, idx, cq, kn, qn, vv, g, beta, o, h, nk,
                         k ** -0.5)
    torch.npu.synchronize()
    _WARM_KEEP = (state, idx, cq, kn, qn, vv, g, beta, o)
    _WARMED = ok
    return ok
