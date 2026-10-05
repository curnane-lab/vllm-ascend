# SPDX-License-Identifier: Apache-2.0
"""AscendC LeapQuant GDN decode step (A3) — ctypes binding wrapper.

Calls the custom opp op ``GdnLqStep`` (gdn_lq_step) in libcust_opapi.so: one
kernel per decode step, in-place window append inside the fp32 ssm_state slot
buffer plus a bf16 output tensor. Modeled on gdn_lq_flush_ascendc.py:
IgnoreContiguous strided state, one shared workspace, executor cache keyed by
data pointers, clean False fallback while a graph capture is recording.
"""

import ctypes
import glob
import os

import torch

_B = None
_AVAILABLE = None
_CACHE = {}
_CACHE_CAP = 256  # keyed by data_ptrs; addresses are stable in decode loops
_WS_BUF = None  # one shared workspace: the op's kernel takes no workspace
# argument, and aclnn executions on the same stream are serialized, so a
# single buffer sized to the max requested workspace serves all executors.


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
        lib.aclnnGdnLqStepGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnLqStep.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.aclnnGdnLqStep.restype = ctypes.c_int
    return _B


def gdn_lq_step_ascendc(state, idx, kn, qn, v, g_log, beta, o, num_heads,
                        scale) -> bool:
    """One decode step through the AscendC kernel; returns False when the
    custom op is unavailable (caller falls back to the triton/torch path).

    state: fp32 [SLOTS, H, V, K] ssm_state buffer (strided view is fine).
    kn/qn: bf16 [T, H, K] l2-normalized (qn unscaled); v: bf16 [T, H, V];
    g_log: fp32 [T, H]; beta: bf16 [T, H]; o: bf16 [T, H, V] output.
    """
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

    def _c(x, dt):
        x = x.to(device=dev, dtype=dt)
        return x if x.is_contiguous() else x.contiguous()

    idx = _c(idx, torch.int64)
    kn = _c(kn, torch.bfloat16)
    qn = _c(qn, torch.bfloat16)
    v = _c(v, torch.bfloat16)
    g_log = _c(g_log, torch.float32)
    beta = _c(beta, torch.bfloat16)
    slot_stride_bytes = state.stride(0) * state.element_size()
    stream = int(torch.npu.current_stream().npu_stream)
    key = (int(state.data_ptr()), int(idx.data_ptr()), int(kn.data_ptr()),
           int(qn.data_ptr()), int(v.data_ptr()), int(g_log.data_ptr()),
           int(beta.data_ptr()), int(o.data_ptr()), int(state.stride(0)),
           int(idx.shape[0]), int(num_heads), float(scale))
    hit = _CACHE.get(key)
    global _WS_BUF
    if hit is not None:
        executor = hit[0]
        ret = lib.aclnnGdnLqStep(
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
            ctx.tensor(state, "state"),
            ctx.tensor(idx, "idx"),
            ctx.tensor(kn, "kn"),
            ctx.tensor(qn, "qn"),
            ctx.tensor(v, "v"),
            ctx.tensor(g_log, "g_log"),
            ctx.tensor(beta, "beta"),
            ctypes.c_int64(int(num_heads)),
            ctypes.c_int64(int(slot_stride_bytes)),
            ctypes.c_double(float(scale)),
            ctx.tensor(o, "out"),
        ]
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnLqStepGetWorkspaceSize(
            *args, ctypes.byref(ws_size), ctypes.byref(executor))
        assert ret == 0, f"get_ws ret={ret} ws={ws_size.value}"
        need = int(ws_size.value) if ws_size.value else 1
        if _WS_BUF is None or _WS_BUF.numel() < need:
            _WS_BUF = torch.empty(need, dtype=torch.uint8, device=dev)
        ret = lib.aclnnGdnLqStep(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(need), executor,
            ctypes.c_void_p(stream))
        assert ret == 0, f"launch ret={ret}"
        if len(_CACHE) >= _CACHE_CAP:
            _CACHE.clear()
        # keep the tensors alive: the executor bakes their addresses
        _CACHE[key] = (executor, state, idx, kn, qn, v, g_log, beta, o)
        return True
    except Exception as e:
        import sys
        print(f"[gdn-lq-step] slow-path exception: {type(e).__name__}: {e!r}",
              file=sys.stderr, flush=True)
        return False
    finally:
        ctx.destroy()


def step_pools_ascendc(pools, idx, kn, qn, v, g_log, beta, scale, o=None):
    """Drop-in replacement for GDNLeapQuantPools.decode_step; returns o."""
    if o is None:
        o = torch.empty(idx.shape[0], pools.num_heads, pools.head_v,
                        dtype=torch.bfloat16, device=idx.device)
    ok = gdn_lq_step_ascendc(_state_tensor(pools), idx, kn, qn, v, g_log,
                             beta, o, pools.num_heads, scale)
    if not ok:
        return None
    return o


def _state_tensor(pools):
    # pools keeps the strided int8 view over the layer ssm_state; recover the
    # fp32 view with the same slot stride so the kernel sees identical layout
    flat = pools._flat  # [num_slots, slot_bytes] int8
    return flat.view(torch.float32)
