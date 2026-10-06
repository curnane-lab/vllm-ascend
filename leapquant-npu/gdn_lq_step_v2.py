"""M2-D2 step v2: three-op decode step pipeline (dequant -> cube GEMV ->
epilogue), stream-ordered. Drop-in replacement for the triton step /
pools.decode_step hot path.

Production-form input pooling: the binding executor cache keys embed
data_ptrs, so passing freshly-allocated per-step kn/qn/v/g_log/beta/o
misses the cache every step and pays the slow path (host sync) three
times (~5-29 ms/step). This module keeps a small ring of pre-allocated
input slots and copy_()s each step's data into the fixed slots, keeping
every executor key constant after the first len(ring) steps. Ring depth
4 covers the callers that hold the returned o for a few steps (gdn.py
consumes it into core_attn_out immediately); raise _RING if needed.
The copies are async stream ops (~30-50 us total, measured in the
M2-D2 report).
"""

import os as _os

import torch

from gdn_lq_dequant import gdn_lq_dequant, warm_cache as warm_dequant
from gdn_lq_epilogue import gdn_lq_epilogue, warm_cache as warm_epi
from gdn_lq_gemv import gdn_lq_gemv, warm_cache as warm_gemv

_BUF_CACHE = {}
# Single slot per (t, nk, h): the aclnn runtime executor pool keys on the
# op type + tensor-desc shape, NOT on data_ptrs, so creating a second
# executor for the same op (e.g. a 4-slot ring) silently reuses the
# internal block and corrupts the first executor - hits then launch the
# wrong plan (found 2026-10-06: ring-of-4 reuse broke o at step 5 while
# ring-of-1 is exact for 96+ steps). One executor per op per shape, hit
# on every step after the first; the copies are stream-ordered after the
# previous step's kernels, so serial decode steps lose nothing.
# Consequence: the returned o is a pool view valid only until the next
# step_pools_v2 call with the same (t, nk, h) - gdn.py consumes it into
# core_attn_out immediately; snapshot with .cpu() if holding longer.
_RING = 1
_POOL = {}


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


def _make_slot(t, nk, h, v, k, dev):
    return {
        "kn": torch.empty(t, nk, k, dtype=torch.bfloat16, device=dev),
        "qn": torch.empty(t, nk, k, dtype=torch.bfloat16, device=dev),
        "v": torch.empty(t, h, v, dtype=torch.bfloat16, device=dev),
        "g_log": torch.empty(t, h, dtype=torch.float32, device=dev),
        "beta": torch.empty(t, h, dtype=torch.bfloat16, device=dev),
        "o": torch.empty(t, h, v, dtype=torch.bfloat16, device=dev),
    }


def _slot(t, nk, h, v, k, dev):
    """Round-robin input slot with stable data_ptrs (executor key hit)."""
    key = (t, nk, h, str(dev))
    ent = _POOL.get(key)
    if ent is None:
        ent = [[_make_slot(t, nk, h, v, k, dev) for _ in range(_RING)], 0]
        if len(_POOL) > 32:
            _POOL.clear()
        _POOL[key] = ent
    slots = ent[0]
    slot = slots[ent[1] % _RING]
    ent[1] += 1
    return slot


def warm_step_v2(device="npu"):
    return (warm_dequant(device) and warm_gemv(device) and warm_epi(device))


def step_pools_v2(pools, idx, kn, qn, v, g_log, beta, scale, o=None):
    """Full decode-step contract (same as GDNLeapQuantPools.decode_step):
    kn/qn bf16 [T, nk, K] l2-normalized, UNexpanded key-head rows; v bf16
    [T, H, V]; g_log fp32 [T, H]; beta bf16 [T, H]; idx int64 [T].
    Returns o bf16 [T, H, V] (a ring-slot view: valid until the same
    slot is reused _RING steps later) or None when unavailable.
    In-place window append inside pools state, same as v1."""
    h = pools.num_heads
    nk = kn.shape[1]
    t = idx.shape[0]
    dev = idx.device
    slot = _slot(t, nk, h, pools.head_v, pools.head_k, dev)
    # async device-side copies into the fixed slots; data_ptrs constant
    slot["kn"].copy_(kn)
    slot["qn"].copy_(qn)
    slot["v"].copy_(v)
    slot["g_log"].copy_(g_log)
    slot["beta"].copy_(beta)
    kn_s = slot["kn"]
    qn_s = slot["qn"]
    v_s = slot["v"]
    g_s = slot["g_log"]
    b_s = slot["beta"]
    o_s = slot["o"] if o is None else o
    dbg = _os.getenv("VLLM_ASCEND_GDN_LQ_DEBUG")

    def _mem(tag):
        if dbg:
            print(f"[lq-v2] {tag} ok_alloc={torch.npu.memory_allocated()/2**30:.2f}GiB",
                  flush=True)
    staging, aq, cq = _bufs(t, h, pools.head_v, pools.head_k, dev)
    ok = gdn_lq_dequant(pools._state, idx, kn_s, qn_s, staging, aq, h, nk,
                        scale)
    _mem(f"dequant ret={ok}")
    ok = ok and gdn_lq_gemv(staging, aq, cq)
    _mem(f"gemv ret={ok}")
    ok = ok and gdn_lq_epilogue(pools._state, idx, cq, kn_s, qn_s, v_s, g_s,
                                b_s, o_s, h, nk, scale)
    _mem(f"epilogue ret={ok}")
    if not ok:
        return None
    return o_s
