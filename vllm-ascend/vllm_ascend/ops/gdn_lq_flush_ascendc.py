# SPDX-License-Identifier: Apache-2.0
"""AscendC LeapQuant GDN window flush (A3) — ctypes binding wrapper.

Calls the custom opp op ``GdnLqFlush`` (gdn_lq_flush) in libcust_opapi.so.
The op updates the LeapQuant composite checkpoint in place inside the fp32
ssm_state slot buffer; the caller passes the same tensor as output.

q0 is expected pre-transposed as [R, V] fp32 (row r = torch q0[:, r]).
"""

import ctypes
import glob
import os
import sys

import torch

_B = None
_AVAILABLE = None
_CACHE = {}
_CACHE_CAP = 256  # keyed by data_ptrs; addresses are stable in decode loops
_HITS = 0
_NEW_EXECUTORS = 0
_FALLBACKS = 0
_WS_BUF = None  # one shared workspace: the op's kernel takes no workspace
# argument, and aclnn executions on the same stream are serialized, so a
# single buffer sized to the max requested workspace serves all executors
# (the autogen wrapper derives ~state-size workspace per call site).


def _flush_lib_candidates():
    """Search every known opp install. The first file on disk is not
    necessarily the build that exports GdnLqFlush: an older libcust_opapi.so
    makes GetWorkspaceSize return 161001 and the e2e path falls back to
    torch.flush."""
    pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    patterns = [
        "/tmp/fla_npu_install/fla_npu/opp/vendors/*/op_api/lib/libcust_opapi.so",
        "/tmp/probe_opp_v*/vendors/*/op_api/lib/libcust_opapi.so",
        "/usr/local/Ascend/opp/vendors/*/op_api/lib/libcust_opapi.so",
        os.path.join(pkg, "_cann_ops_custom", "**", "op_api", "lib",
                     "libcust_opapi.so"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))
    return found


def _load_flush_lib(paths):
    rejected = []
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            rejected.append(f"{path}: {exc}")
            continue
        if not hasattr(lib, "aclnnGdnLqFlushGetWorkspaceSize"):
            rejected.append(path + ": missing aclnnGdnLqFlushGetWorkspaceSize")
            continue
        return lib, path, rejected
    return None, None, rejected


def _binding():
    global _B, _AVAILABLE
    if _AVAILABLE is False:
        return None
    if _B is None:
        lib, path, rejected = _load_flush_lib(_flush_lib_candidates())
        if lib is None:
            _AVAILABLE = False
            if rejected:
                print("[gdn-lq-flush] no libcust_opapi.so exports "
                      "aclnnGdnLqFlushGetWorkspaceSize: "
                      + "; ".join(rejected),
                      file=sys.stderr, flush=True)
            return None
        print(f"[gdn-lq-flush] loaded {path}", file=sys.stderr, flush=True)
        import torch_npu  # noqa: F401

        from fla_npu.ops.ascendc import _runtime as rt

        _B = (rt, rt.runtime(), lib)
        lib.aclnnGdnLqFlushGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnLqFlush.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        lib.aclnnGdnLqFlush.restype = ctypes.c_int
    return _B


def gdn_lq_flush_ascendc(state, idx, q0_rv, num_heads) -> bool:
    """Flush due slots of the decode batch idx in place.

    state: fp32 [SLOTS, H, V, K] ssm_state buffer (strided view is fine, the
        slot stride in bytes is taken from state.stride(0) * 4).
    idx: int64 [T] slot ids.
    q0_rv: fp32 [R, V] subspace seed, rows = torch q0 columns.
    Returns False when the custom op is unavailable (caller falls back).
    """
    try:
        capturing = torch.npu.is_current_stream_capturing()
    except Exception:
        capturing = False
    global _HITS, _NEW_EXECUTORS, _FALLBACKS
    bnd = _binding()
    if bnd is None:
        _FALLBACKS += 1
        return False
    rt, runtime, lib = bnd
    dev = state.device
    idx = idx.to(device=dev, dtype=torch.int64)
    if not idx.is_contiguous():
        idx = idx.contiguous()
    if not q0_rv.is_contiguous():
        q0_rv = q0_rv.contiguous()
    slot_stride_bytes = state.stride(0) * state.element_size()
    stream = int(torch.npu.current_stream().npu_stream)
    key = (int(state.data_ptr()), int(idx.data_ptr()), int(q0_rv.data_ptr()),
           int(state.stride(0)), int(idx.shape[0]), int(num_heads))
    hit = _CACHE.get(key)
    global _WS_BUF
    if hit is not None:
        if capturing and _WS_BUF is None:
            _FALLBACKS += 1
            return False
        executor = hit[0]
        ret = lib.aclnnGdnLqFlush(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(_WS_BUF.numel()),
            executor, ctypes.c_void_p(stream))
        if ret == 0:
            _HITS += 1
            return True
        _FALLBACKS += 1
        return False
    if capturing:
        # cannot build aclTensor descriptors / executors mid-capture
        _FALLBACKS += 1
        return False
    try:
        ctx = rt._CallContext(runtime, dev)
    except Exception:
        _FALLBACKS += 1
        return False
    try:
        args = [
            # pass a 1-element view of the pool base: the executor would
            # otherwise materialize the whole (giant, strided) ssm_state
            # view at creation (~2.6 GiB per executor, leaking one copy per
            # step in decode loops with fresh executor keys). The kernel
            # addresses slots through the base pointer + s_slot stride and
            # never reads the tensor body. as_strided avoids the
            # contiguous() copy that _c would trigger.
            ctx.tensor(torch.as_strided(state, (1,), (1,)), "state"),
            ctx.tensor(idx, "idx"),
            ctx.tensor(q0_rv, "q0"),
            ctypes.c_int64(int(num_heads)),
            ctypes.c_int64(int(slot_stride_bytes)),
            # in-place alias of the same pool: base view for the same reason
            ctx.tensor(torch.as_strided(state, (1,), (1,)), "out"),
        ]
        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        ret = lib.aclnnGdnLqFlushGetWorkspaceSize(
            *args, ctypes.byref(ws_size), ctypes.byref(executor))
        assert ret == 0, (
            f"get_ws ret={ret} ws={ws_size.value} "
            f"state_shape={tuple(state.shape)} state_stride0={state.stride(0)} "
            f"slot_stride_bytes={slot_stride_bytes} num_heads={num_heads} "
            f"idx_shape={tuple(idx.shape)} idx_dtype={idx.dtype} "
            f"q0_shape={tuple(q0_rv.shape)}"
        )
        if os.getenv("VLLM_ASCEND_GDN_LQ_DEBUG"):
            print(
                f"[gdn-lq-flush] new executor key_state={int(state.data_ptr()) & 0xffff} "
                f"T={idx.shape[0]} ws={ws_size.value} "
                f"allocated={torch.npu.memory_allocated() / 2**30:.2f}GiB",
                flush=True,
            )
        need = int(ws_size.value) if ws_size.value else 1
        if _WS_BUF is None or _WS_BUF.numel() < need:
            _WS_BUF = torch.empty(need, dtype=torch.uint8, device=dev)
        ret = lib.aclnnGdnLqFlush(
            ctypes.c_void_p(int(_WS_BUF.data_ptr())),
            ctypes.c_uint64(need), executor,
            ctypes.c_void_p(stream))
        assert ret == 0, f"launch ret={ret}"
        if len(_CACHE) >= _CACHE_CAP:
            _CACHE.clear()
        # keep the tensors alive: the executor bakes their addresses
        _CACHE[key] = (executor, state, idx, q0_rv)
        _NEW_EXECUTORS += 1
        return True
    except Exception as e:
        import sys
        _FALLBACKS += 1
        print(f"[gdn-lq-flush] slow-path exception: {type(e).__name__}: {e!r}",
              file=sys.stderr, flush=True)
        return False
    finally:
        ctx.destroy()


def flush_pools_ascendc(pools, idx) -> bool:
    """Drop-in replacement for GDNLeapQuantPools.flush via the AscendC op."""
    return gdn_lq_flush_ascendc(_state_tensor(pools), idx,
                                _q0_rows(pools), pools.num_heads)


_Q0_ROWS = {}


def _q0_rows(pools):
    """Persistent [R, V] transpose of pools.q0 (stable executor cache key)."""
    key = (pools.q0.device, int(pools.q0.data_ptr()), tuple(pools.q0.shape))
    buf = _Q0_ROWS.get(key)
    if buf is None:
        buf = pools.q0.t().contiguous()
        _Q0_ROWS.clear()  # bounded: one entry per pools instance
        _Q0_ROWS[key] = buf
    return buf


def _state_tensor(pools):
    # the fp32 ssm_state view the pools were built on; the kernel addresses
    # slots through the s_slot byte-stride attribute, heads through HEAD_BYTES
    return pools._state


def flush_counters():
    """Executor-cache counters for the AscendC flush binding.

    ``calls`` counts every entry, including fallbacks. A newly built
    executor is not also counted as a hit.
    """
    return {
        "calls": _HITS + _NEW_EXECUTORS + _FALLBACKS,
        "hits": _HITS,
        "new_executors": _NEW_EXECUTORS,
        "fallbacks": _FALLBACKS,
    }
