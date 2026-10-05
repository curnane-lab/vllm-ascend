"""M2-D1 probe: AscendC cube-GEMV LeapQuant core (A3) — ctypes binding.

Calls the custom opp op ``GdnLqGemvProbe`` (gdn_lq_gemv_probe) in
libcust_opapi.so: one fp16 16x128x128 GEMM per instance on the cube core,
C fp32 [G,16,128] (rows [n,16) are zero-pad products, slice to [G,n,128]).
Modeled on gdn_lq_step_ascendc.py: executor cache keyed by data pointers.
"""

import ctypes
import glob
import os

import torch

# the aclnn executor resolves kernel binaries through the opp vendor
# package; make sure the built package is registered before the first op
# call (mirrors vllm_ascend.utils.bootstrap_custom_op_env, without the
# heavy vllm import — must run before aclnnGetOpExecutor registration)
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
        lib.aclnnGdnLqGemvProbeGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnLqGemvProbe.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.aclnnGdnLqGemvProbe.restype = ctypes.c_int
    return _B


def gdn_lq_gemv_probe(a, b, c) -> bool:
    """C[g] = A[g] @ B[g] on the cube: fp16 [G,16,128] x fp16 [G,128,128]
    -> fp32 [G,16,128]. Returns False when the custom op is unavailable."""
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except Exception:
        return False
    bnd = _binding()
    if bnd is None:
        return False
    rt, runtime, lib = bnd
    dev = a.device

    def _c(x, dt):
        x = x.to(device=dev, dtype=dt)
        return x if x.is_contiguous() else x.contiguous()

    a = _c(a, torch.float16)
    b = _c(b, torch.float16)
    stream = int(torch.npu.current_stream().npu_stream)
    key = (int(a.data_ptr()), int(b.data_ptr()), int(c.data_ptr()),
           int(a.shape[0]))
    hit = _CACHE.get(key)
    global _WS_BUF
    if hit is not None:
        executor = hit[0]
        ret = lib.aclnnGdnLqGemvProbe(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(_WS_BUF.numel()),
            executor, ctypes.c_void_p(stream))
        return ret == 0
    try:
        ctx = rt._CallContext(runtime, dev)
    except Exception:
        return False
    try:
        args = [
            ctx.tensor(a, "a"),
            ctx.tensor(b, "b"),
            ctx.tensor(c, "c"),
        ]
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnLqGemvProbeGetWorkspaceSize(
            *args, ctypes.byref(ws_size), ctypes.byref(executor))
        assert ret == 0, f"get_ws ret={ret} ws={ws_size.value}"
        need = int(ws_size.value) if ws_size.value else 1
        if _WS_BUF is None or _WS_BUF.numel() < need:
            _WS_BUF = torch.empty(need, dtype=torch.uint8, device=dev)
        ret = lib.aclnnGdnLqGemvProbe(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(need), executor,
            ctypes.c_void_p(stream))
        assert ret == 0, f"launch ret={ret}"
        if len(_CACHE) >= _CACHE_CAP:
            _CACHE.clear()
        # keep the tensors alive: the executor bakes their addresses
        _CACHE[key] = (executor, a, b, c)
        return True
    except Exception as e:
        import sys
        print(f"[gdn-lq-gemv-probe] slow-path exception: "
              f"{type(e).__name__}: {e!r}", file=sys.stderr, flush=True)
        return False
    finally:
        ctx.destroy()
