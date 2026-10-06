"""M2-D2 step v2: three-op decode step pipeline (dequant -> cube GEMV ->
epilogue), stream-ordered. Drop-in replacement for the triton step /
pools.decode_step hot path."""

import torch

from gdn_lq_dequant import gdn_lq_dequant, warm_cache as warm_dequant
from gdn_lq_epilogue import gdn_lq_epilogue, warm_cache as warm_epi
from gdn_lq_gemv import gdn_lq_gemv, warm_cache as warm_gemv

_BUF_CACHE = {}


def _bufs(t, h, v, k, dev):
    key = (t, h, dev)
    hit = _BUF_CACHE.get(key)
    if hit is not None:
        return hit
    bufs = (
        torch.empty(t, h, v, k, dtype=torch.float16, device=dev),
        torch.empty(t, h, 16, k, dtype=torch.float16, device=dev),
        torch.empty(t, h, 16, v, dtype=torch.float32, device=dev),
    )
    if len(_BUF_CACHE) > 64:
        _BUF_CACHE.clear()
    _BUF_CACHE[key] = bufs
    return bufs


def warm_step_v2(device="npu"):
    return (warm_dequant(device) and warm_gemv(device) and warm_epi(device))


def step_pools_v2(pools, idx, kn, qn, v, g_log, beta, scale, o=None):
    """Full decode-step contract (same as GDNLeapQuantPools.decode_step):
    kn/qn bf16 [T, nk, K] l2-normalized, UNexpanded key-head rows; v bf16
    [T, H, V]; g_log fp32 [T, H]; beta bf16 [T, H]; idx int64 [T].
    Returns o bf16 [T, H, V] or None when unavailable. In-place window
    append inside pools state, same as v1."""
    h = pools.num_heads
    nk = kn.shape[1]
    t = idx.shape[0]
    dev = idx.device
    if o is None:
        o = torch.empty(t, h, pools.head_v, dtype=torch.bfloat16, device=dev)
    import os as _os
    if _os.getenv("VLLM_ASCEND_GDN_LQ_DEBUG"):
        print(f"[lq-v2] step t={t} h={h} mem_alloc={torch.npu.memory_allocated()/2**30:.2f}GiB "
              f"reserved={torch.npu.memory_reserved()/2**30:.2f}GiB", flush=True)
    staging, aq, cq = _bufs(t, h, pools.head_v, pools.head_k, dev)
    import os as _os
    dbg = _os.getenv("VLLM_ASCEND_GDN_LQ_DEBUG")
    def _mem(tag):
        if dbg:
            print(f"[lq-v2] {tag} ok_alloc={torch.npu.memory_allocated()/2**30:.2f}GiB", flush=True)
    ok = gdn_lq_dequant(pools._state, idx, kn, qn, staging, aq, h, nk, scale)
    _mem(f"dequant ret={ok}")
    ok = ok and gdn_lq_gemv(staging, aq, cq)
    _mem(f"gemv ret={ok}")
    ok = ok and gdn_lq_epilogue(pools._state, idx, cq, kn, qn, v, g_log,
                                beta, o, h, nk, scale)
    _mem(f"epilogue ret={ok}")
    if not ok:
        return None
    return o
