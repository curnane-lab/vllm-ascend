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

from gdn_lq_dequant import gdn_lq_dequant, launch as dequant_launch
from gdn_lq_dequant import warm_cache as warm_dequant
from gdn_lq_epilogue import gdn_lq_epilogue, launch as epi_launch
from gdn_lq_epilogue import warm_cache as warm_epi
from gdn_lq_gemv import gdn_lq_gemv, launch as gemv_launch
from gdn_lq_gemv import warm_cache as warm_gemv

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
        "idx": torch.empty(t, dtype=torch.int64, device=dev),
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


def step_pools_v2_fused(pools, idx, kn, qn, v, g_log, beta, scale):
    """E1 fused entry: one host call enqueues dequant -> gemv -> epilogue.

    Single stream-capture check, single stream fetch, single binding
    resolve for the whole step (the per-op wrappers redo these, which at
    24 layers x 3 ops is measurable host time). idx joins the input pool:
    gdn.py's per-layer ``indices.flatten().long()`` produced a fresh
    tensor whose data_ptr is part of the executor cache key; the pooled
    slot keeps the key constant (E1). idx may be int32/int64, any layout
    (the slot copy_ casts and densifies, async on the same stream).

    Same contract as step_pools_v2: returns o bf16 [T,H,V] as a pool view
    valid until the next call with the same (t, nk, h); None -> caller
    falls back to the triton step.
    """
    try:
        if torch.npu.is_current_stream_capturing():
            return None
    except Exception:
        return None
    h = pools.num_heads
    nk = kn.shape[1]
    t = kn.shape[0]
    dev = kn.device
    slot = _slot(t, nk, h, pools.head_v, pools.head_k, dev)
    # idx stays a fresh .long() cast passed DIRECTLY to the kernels:
    # that is the only arrangement with correct engine-context numerics
    # (48-token smoke 3/4 @token 36). Pooling idx into the fixed slot
    # keeps the executor cache key constant and is ~2x faster e2e, but
    # deterministically corrupts slot addressing in the real gdn.py
    # pipeline (diverges @token 2) while every isolated harness - sync'd
    # or async, single- or multi-pools, layer-multiplexed - is bit-exact
    # and the slot memory verifies intact around the kernels. Root cause
    # unresolved; the A/B switch (VLLM_ASCEND_GDN_LQ_IDXPOOL=1) plus
    # VLLM_ASCEND_GDN_LQ_IDXWATCH=1 keep the experiment reproducible.
    # E1 exit gate with this arrangement: B=256 e2e 389 ms/step >= 279
    # -> step v2 stays behind its default-off flag.
    if _os.getenv("VLLM_ASCEND_GDN_LQ_IDXPOOL"):
        slot["idx"].copy_(idx.reshape(-1).long())
        idx_s = slot["idx"]
    else:
        idx_s = idx.reshape(-1).long()
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
    o_s = slot["o"]
    staging, aq, cq = _bufs(t, h, pools.head_v, pools.head_k, dev)
    if _os.getenv("VLLM_ASCEND_GDN_LQ_NOFUSE"):
        # A/B switch: public per-op wrappers + unpooled idx (exact pre-E1
        # semantics) to isolate the fused-entry delta.
        idx_u = idx if _os.getenv("VLLM_ASCEND_GDN_LQ_NOPOOLIDX") else idx_s
        ok = gdn_lq_dequant(pools._state, idx_u, kn_s, qn_s, staging, aq, h,
                            nk, scale)
        ok = ok and gdn_lq_gemv(staging, aq, cq)
        ok = ok and gdn_lq_epilogue(pools._state, idx_u, cq, kn_s, qn_s,
                                    v_s, g_s, b_s, o_s, h, nk, scale)
        return o_s if ok else None
    stream = int(torch.npu.current_stream().npu_stream)
    ok = dequant_launch(pools._state, idx_s, kn_s, qn_s, staging, aq, h,
                        nk, scale, stream)
    ok = ok and gemv_launch(staging, aq, cq, stream)
    ok = ok and epi_launch(pools._state, idx_s, cq, kn_s, qn_s, v_s, g_s,
                           b_s, o_s, h, nk, scale, stream)
    if not ok:
        return None
    if _os.getenv("VLLM_ASCEND_GDN_LQ_IDXWATCH"):
        global _IDXW
        try:
            _IDXW += 1
        except NameError:
            _IDXW = 0
        if _IDXW < 12:
            expect = idx.reshape(-1).long().cpu()
            torch.npu.synchronize()
            got = slot["idx"].cpu()
            print(f"[idxwatch] call {_IDXW} post-kernel intact="
                  f"{bool((expect == got).all())} "
                  f"expect={expect[:6].tolist()} got={got[:6].tolist()}",
                  flush=True)
    return o_s


def step_pools_v2(pools, idx, kn, qn, v, g_log, beta, scale, o=None):
    """Full decode-step contract (same as GDNLeapQuantPools.decode_step):
    kn/qn bf16 [T, nk, K] l2-normalized, UNexpanded key-head rows; v bf16
    [T, H, V]; g_log fp32 [T, H]; beta bf16 [T, H]; idx int64 [T].
    Returns o bf16 [T, H, V] (a ring-slot view: valid until the same
    slot is reused _RING steps later) or None when unavailable.
    In-place window append inside pools state, same as v1."""
    return step_pools_v2_fused(pools, idx, kn, qn, v, g_log, beta, scale)
