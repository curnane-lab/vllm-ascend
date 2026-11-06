#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""AscendC GDN indexed state-pool prefill.

Calls the custom op ``GdnPoolPrefill`` in ``libcust_opapi.so``. One launch
reads initial recurrent states from source pool slots and writes final states
to destination pool slots. ``None`` means the caller should keep the compact
gather/scatter path.

The loaded library handle is process-lifetime state. The first caller searches
once; a miss stays a miss until the process restarts. Later calls only read
that cache.
"""

from __future__ import annotations

import ctypes
import os

import torch

_OPAPI_NAME = "libcust_opapi.so"
_PROTOTYPE_OPAPI = "/tmp/gdn_opp_install/vendors/custom_transformer/op_api/lib/" + _OPAPI_NAME
_FUSED_CHUNK_SIZE = 64

# Process-lifetime cache of (runtime module, runtime, CDLL), or unset.
_BINDING = None
_AVAILABLE: bool | None = None


def _opapi_candidates() -> list[str]:
    """Return installed opapi libraries that may export ``GdnPoolPrefill``."""
    raw_paths = [_PROTOTYPE_OPAPI]
    custom_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    for entry in custom_opp.split(":"):
        if not entry:
            continue
        raw_paths.append(os.path.join(entry, "op_api", "lib", _OPAPI_NAME))
        raw_paths.append(os.path.join(entry, "vendors", "custom_transformer", "op_api", "lib", _OPAPI_NAME))
    found: list[str] = []
    for path in raw_paths:
        if path not in found and os.path.isfile(path):
            found.append(path)
    return found


def _binding():
    global _BINDING, _AVAILABLE
    if _AVAILABLE is False:
        return None
    if _BINDING is not None:
        return _BINDING
    for path in _opapi_candidates():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        if not hasattr(lib, "aclnnGdnPoolPrefillGetWorkspaceSize") or not hasattr(lib, "aclnnGdnPoolPrefill"):
            continue
        try:
            import torch_npu  # noqa: F401

            from fla_npu.ops.ascendc import _runtime as rt
        except ImportError:
            _AVAILABLE = False
            return None
        lib.aclnnGdnPoolPrefillGetWorkspaceSize.restype = ctypes.c_int
        lib.aclnnGdnPoolPrefill.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.aclnnGdnPoolPrefill.restype = ctypes.c_int
        _BINDING = (rt, rt.runtime(), lib)
        _AVAILABLE = True
        return _BINDING
    _AVAILABLE = False
    return None


def _capturing() -> bool:
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_current_stream_capturing())
    except Exception:
        # Graph-capture probes differ across CANN builds. Skip the custom op
        # rather than failing the prefill launch.
        return True


def gdn_pool_prefill_ascendc(
    src_pool: torch.Tensor,
    dst_pool: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    has_initial_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor | None:
    """Run indexed state-pool GDN prefill, or return None so the caller can fall back.

    Pools stay as strided views. Their slot strides are passed to the kernel, so
    this does not compact the whole cache. Token inputs are packed contiguously.

    src_pool: bf16 [SLOTS, Hv, Dv, Dk] source state pool.
    dst_pool: bf16 same rank; may alias src_pool for an in-place slot update.
    src_slots: int32 [B] source pool slot indices.
    dst_slots: int32 [B] destination pool slot indices.
    has_initial_state: bool [B].
    q/k: bf16 [B, Hk, T, Dk] BNSD.
    v: bf16 [B, Hv, T, Dv] BNSD.
    g: fp32 [B, T, Hv] BTH.
    beta: bf16 [B, T, Hv] BTH.

    Returns:
        output bf16 [B, T, Hv, Dv] BSND, or None when the op cannot run.
    """
    if _capturing():
        return None
    bound = _binding()
    if bound is None:
        return None
    rt, _runtime, lib = bound
    device = src_pool.device

    def _as(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor.to(device=device, dtype=dtype).contiguous()

    src_slots_c = _as(src_slots, torch.int32)
    dst_slots_c = _as(dst_slots, torch.int32)
    has_init_c = _as(has_initial_state, torch.bool)
    q_c = _as(q, torch.bfloat16)
    k_c = _as(k, torch.bfloat16)
    v_c = _as(v, torch.bfloat16)
    g_c = _as(g, torch.float32)
    beta_c = _as(beta, torch.bfloat16)
    out = torch.empty(
        q_c.shape[0],
        v_c.shape[2],
        v_c.shape[1],
        v_c.shape[3],
        dtype=torch.bfloat16,
        device=device,
    )
    src_slot_stride = src_pool.stride(0) * src_pool.element_size()
    dst_slot_stride = dst_pool.stride(0) * dst_pool.element_size()
    try:
        ret = rt.call(
            lib.aclnnGdnPoolPrefillGetWorkspaceSize,
            lib.aclnnGdnPoolPrefill,
            src_pool,
            dst_pool,
            src_slots_c,
            dst_slots_c,
            has_init_c,
            q_c,
            k_c,
            v_c,
            g_c,
            beta_c,
            out,
            ctypes.c_float(scale),
            ctypes.c_int64(_FUSED_CHUNK_SIZE),
            ctypes.c_int64(src_slot_stride),
            ctypes.c_int64(dst_slot_stride),
        )
    except Exception:
        return None
    if ret != 0:
        return None
    return out
