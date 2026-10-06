"""M3-3b flush v2: three-op cube pipeline orchestrator + ctypes bindings.

Chain per flush call (stream-ordered, due = fill >= L scanned in-kernel):
  fprep(AIV) -> fgemv(1: S=A^T B fp16) -> fgemv(2: m1=S^T q0)
  -> ftail(1: qk=mgs(m1)) -> fgemv(3: qv=S qk)
  -> ftail(2: qv'=mgs(qv)) -> fgemv(4: uk=S^T qv')
  -> ftail(3: E/quantize/writeback)
gdn_lq_flush (v1) remains the fallback path.

Due gating without host sync (M3-3b fix): fprep writes each head's fill
VALUE into ``flags``; the wrapper copies flags to pinned memory
(asynchronously) and decides the remaining 7 launches from LAST step's
pinned values: a head is due this step iff its fill was L-1 last step
(decode advances fill by exactly one per step; resync resets it to 0).
Two consecutive fill >= L readings also trigger (flush called without an
intervening decode step - bench loops only; production always decodes,
wrapping fill to 1). Copy completion is proven by a device round-tripped
sequence number (torch.npu.Event.query() is unreliable on this stack),
never by synchronize. Steady-state nothing-due costs one fprep launch
plus two tiny async copies.
"""

import ctypes
import glob
import os

import numpy as np
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
_WS_BUF = None


def _binding():
    global _B
    if _B is None:
        cands = (glob.glob("/tmp/probe_opp_v*/vendors/custom_transformer/"
                           "op_api/lib/libcust_opapi.so")
                 + glob.glob("/usr/local/Ascend/opp/vendors/*/op_api/lib/"
                             "libcust_opapi.so", recursive=True)
                 + glob.glob("/mnt/workspace/vllm-ascend/vllm_ascend/"
                             "_cann_ops_custom/**/op_api/lib/libcust_opapi.so",
                             recursive=True))
        if not cands:
            raise RuntimeError("libcust_opapi.so not found")
        lib = ctypes.CDLL(cands[0])
        import torch_npu  # noqa: F401
        from fla_npu.ops.ascendc import _runtime as rt
        _B = (rt, rt.runtime(), lib)
    return _B


def _create(lib, rt, runtime, dev, name, tensor_args, attrs, tensors):
    """create an executor; the CallContext MUST stay alive while the
    executor is used (it owns the tensor descriptors)."""
    ctx = rt._CallContext(runtime, dev)
    args = []
    for t, nm in tensor_args:
        args.append(ctx.tensor(t, nm))
    for v in attrs:
        args.append(ctypes.c_int64(int(v)))
    for t, nm in tensors[len(tensor_args):]:
        args.append(ctx.tensor(t, nm))
    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()
    get_ws = getattr(lib, f"aclnn{name}GetWorkspaceSize")
    run = getattr(lib, f"aclnn{name}")
    ret = get_ws(*args, ctypes.byref(ws_size), ctypes.byref(executor))
    assert ret == 0, f"{name} get_ws ret={ret}"
    global _WS_BUF
    need = int(ws_size.value) if ws_size.value else 1
    if _WS_BUF is None or _WS_BUF.numel() < need:
        _WS_BUF = torch.empty(need, dtype=torch.uint8, device=dev)
    return (run, executor, ctx)


def _launch(run_executor, stream):
    run, executor, _ctx = run_executor
    ret = run(ctypes.c_void_p(int(_WS_BUF.data_ptr())),
              ctypes.c_uint64(_WS_BUF.numel()), executor,
              ctypes.c_void_p(stream))
    return ret == 0


def _slow_launch(lib, rt, runtime, dev, name, tensor_args, attrs, tensors):
    """create + launch + destroy, every call. REQUIRED for multi-phase
    ops (GdnLqFgemv/GdnLqFtail): the aclnn runtime pools executors by op
    type + tensor descriptors with the ATTRS EXCLUDED from the key, so
    the phases collide on ONE pool slot and the last-created config wins
    the shared handle - caching phase-specific executors silently runs
    the wrong phase's tiling (observed: a cached g1 returned zeros after
    g2's executor was created). run() snapshots the launch at enqueue
    (verified: an in-flight task is unaffected by the next executor
    creation), so recreating between stages is safe."""
    run_ex = _create(lib, rt, runtime, dev, name, tensor_args, attrs, tensors)
    stream = int(torch.npu.current_stream().npu_stream)
    ok = _launch(run_ex, stream)
    run_ex[2].destroy()
    return ok


_CACHE = {}


def _fprep(state_base, idx, ab, flags, ba, h, s_slot, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    stream = int(torch.npu.current_stream().npu_stream)
    hit = _CACHE.get(key)
    if hit is not None:
        return _launch(hit, stream)
    dev = idx.device
    run_ex = _create(lib, rt, runtime, dev, "GdnLqFprep",
                     [(state_base, "state"), (idx, "idx")],
                     [h, s_slot],
                     [(state_base, "state"), (idx, "idx"),
                      (ab, "ab_stage"), (flags, "flags"),
                      (ba, "ba_stage")])
    ok = _launch(run_ex, stream)
    if ok:
        # single-phase op: one executor per descriptor set, cacheable
        _CACHE[key] = run_ex
    return ok


def _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c, h, s_slot,
           phase, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    return _slow_launch(
        lib, rt, runtime, idx.device, "GdnLqFgemv",
        [(state_base, "state"), (idx, "idx"), (s, "s_stage"),
         (ab, "ab_stage"), (flags, "flags"), (ba, "ba_stage"),
         (q0p, "q0_pad"), (kq, "kq_stage"), (vq, "vq_stage")],
        [h, s_slot, phase],
        [(state_base, "state"), (idx, "idx"), (s, "s_stage"),
         (ab, "ab_stage"), (flags, "flags"), (ba, "ba_stage"),
         (q0p, "q0_pad"), (kq, "kq_stage"), (vq, "vq_stage"),
         (c, "c_stage")])


def _ftail(state_base, idx, s, c, vq, kq, h, s_slot, phase, key):
    bnd = _binding()
    rt, runtime, lib = bnd
    return _slow_launch(lib, rt, runtime, idx.device, "GdnLqFtail",
                        [(state_base, "state"), (idx, "idx"),
                         (s, "s_stage"), (c, "c_stage")],
                        [h, s_slot, phase],
                        [(state_base, "state"), (idx, "idx"),
                         (s, "s_stage"), (c, "c_stage"), (vq, "vq_stage"),
                         (kq, "kq_stage")])


_BUF_CACHE = {}
_Q0_CACHE = {}


def _bufs(t, h, dev):
    key = (t, h, str(dev))
    hit = _BUF_CACHE.get(key)
    if hit is None:
        # per-instance c/kq/vq regions are [128,32]: fgemv phases 2-4 run
        # with N=32 (MatmulApiTiling rejects N=16) and ftail stores/reads
        # the full [.,32] staging row. The previous [.,16] allocation was
        # half the addressed size and the OOB stores faulted (ftail
        # phase-2 vq store MTE invalid-GM-address).
        hit = (
            torch.zeros(t, h, 32, 128, dtype=torch.int32, device=dev),
            torch.zeros(t, h, 192, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 192, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 128, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 32, dtype=torch.float32, device=dev),
            torch.zeros(t, h, 128, 32, dtype=torch.float16, device=dev),
            torch.zeros(t, h, 128, 32, dtype=torch.float16, device=dev),
        )
        _BUF_CACHE[key] = hit
    return hit


def _q0_pad(pools):
    key = int(pools.q0.data_ptr())
    hit = _Q0_CACHE.get(key)
    if hit is None:
        # [128,32]: fgemv phases 2-4 address B with the padded N=32
        # geometry; the extra columns are never consumed (ftail reads
        # cols 0..3) but the address range must be mapped.
        q0p = torch.zeros(128, 32, dtype=torch.float16, device=pools.q0.device)
        q0p[:, :4] = pools.q0.half()
        _Q0_CACHE.clear()
        _Q0_CACHE[key] = q0p
        hit = q0p
    return hit


_WINDOW = 16
# max idx-view staleness in decode steps; the skip path refreshes the
# view every _VIEW_LAG steps (a newly admitted slot cannot be due for
# WINDOW-1 steps, so the lag is safe)
_VIEW_LAG = 4
_MIRROR = {}


def _mirror(pools, dev, n):
    """Pinned double buffer + sequence-tag validity for the sync-free due
    gate, keyed BY SLOT ID (E2). torch.npu.Event is unusable on this
    stack, so copy completion is proven by a device round-tripped
    sequence number. Every flush call enqueues an async idx -> pinned
    copy (plus a fills copy on launch calls); the decision consumes the
    freshest PROVEN buffer. fprep additionally writes each instance's
    slot id into flags plane 1, so the host mirror records
    slot -> (fill, gen) instead of batch-position -> fill: a batch
    position swap must not inherit a stale fill."""
    # Per-pools mirrors: each GDN layer flushes its own pools once per
    # decode step, and a full model forward of device work separates
    # consecutive flushes of the SAME layer, so the async idx/seq copies
    # always prove before that layer's next decision. A shared per-device
    # mirror instead sees 24 flushes per step and its proofs lag
    # unpredictably (intermittent missed folds in the unsynced parity
    # flow). Keyed by id(pools); layers hold their pools for life.
    key = (str(dev), id(pools))
    st = _MIRROR.get(key)
    if st is None or st["cap"] < n:
        cap = max(n, 4096)
        pin = [torch.zeros(2 * cap, dtype=torch.int32, pin_memory=True)
               for _ in range(2)]
        pin_idx = [torch.zeros(cap, dtype=torch.int64, pin_memory=True)
                   for _ in range(2)]
        pin_seq = [torch.zeros(1, dtype=torch.int32, pin_memory=True)
                   for _ in range(2)]
        st = {
            "cap": cap,
            "pin": pin,  # [fills cap | slots cap] int32 per buffer
            "pin_idx": pin_idx,
            "pin_seq": pin_seq,
            "dev_seq": torch.zeros(1, dtype=torch.int32, device=dev),
            "host_seq": torch.zeros(1, dtype=torch.int32, pin_memory=True),
            "pin_np": [t.numpy() for t in pin],
            "idx_np": [t.numpy() for t in pin_idx],
            "seq_np": [t.numpy() for t in pin_seq],
            # per buffer: gen of the copy written there, its batch size,
            # whether it carried fills (launch step)
            "buf_gen": [0, 0],
            "buf_n": [0, 0],       # tokens t (idx view length)
            "buf_inst": [0, 0],    # instances n = t*h (fills/slots length)
            "buf_step": [0, 0],
            "buf_launch": [False, False],
            # slot id -> (fill observed, observation step); the due sim
            "fills": {},
            "gen": 0,
            # pools-side resync epoch (init_from_dense bumps it); a
            # change invalidates every recorded fill - the slots may
            # have been recycled with fresh state
            "epoch": -1,
        }
        _MIRROR[key] = st
    return st


def _apply_proven_launch_copies(st, cur_step):
    """Fold freshly-proven launch buffers into the slot->fill dict.

    The recorded step is the DECODE step the copy observed (the caller
    passes one flush per decode step per layer; the dict is shared
    across layers because all layers' pools advance in lockstep)."""
    for w in (0, 1):
        if not st["buf_launch"][w]:
            continue
        g = st["buf_gen"][w]
        if int(st["seq_np"][w][0]) != g:
            continue
        n = st["buf_inst"][w]
        fills = st["pin_np"][w][:n]
        slots = st["pin_np"][w][n:2 * n]
        ws = st["buf_step"][w]
        seen = set()
        for i in range(n):
            seen.add(int(slots[i]))
            st["fills"][int(slots[i])] = (int(fills[i]), ws)
        # drop slots absent from the launch batch: a departed slot's sim
        # can only grow, and nothing would ever refresh it - stale
        # entries would fire phantom launches. Re-entering slots read as
        # unknown -> conservative due, which is safe because production
        # re-entries always come with a low fill (resync after prefill;
        # decode continuity across a batch removal is impossible).
        for k in [k for k in st["fills"] if k not in seen]:
            del st["fills"][k]
        st["buf_launch"][w] = False
    # drop entries not refreshed recently; a reappearing slot then reads
    # as unknown -> conservative due
    keep = cur_step - 2 * _WINDOW
    for k in [k for k, (_, g) in st["fills"].items() if g < keep]:
        del st["fills"][k]


def launch_stats():
    return {k[1]: v.get("launches", 0) for k, v in _MIRROR.items()}


def reset_due_gate(dev=None):
    """Drop the mirror (next flush refreshes it). For tests that advance
    the window outside the one-decode-per-flush production pattern."""
    if dev is None:
        _MIRROR.clear()
    else:
        _MIRROR.pop(str(dev), None)


def flush_pools_v2(pools, idx, step=None) -> bool:
    """Cube-pipeline flush (M3-3b). Returns False when unavailable."""
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except Exception:
        return False
    try:
        h = pools.num_heads
        t = idx.shape[0]
        dev = idx.device
        n = t * h
        # Sync-free due gate, keyed BY SLOT (E2). Invariants: (1) one
        # decode step per flush call (gdn.py calls this once per layer
        # after the decode step); (2) the caller consumes per-step
        # results before the next flush (vLLM sampling D2H does this in
        # production) so the async proof copies land between flushes.
        # The mirror is per-pools: each layer's flushes are one model
        # forward apart, which is what makes (2) hold per layer. fill advances by exactly one per decode step,
        # wraps to 1 after a fold (fill == L), resync resets it to 0.
        # Every call enqueues an async idx -> pinned copy so the decision
        # sees the batch's slot ids (<=2 steps stale; a guard bounds the
        # staleness, and a slot admitted within the grace window cannot
        # be due yet). Launch iff any viewed slot is UNKNOWN (never
        # observed) or its simulated fill reaches L. Launching is always
        # safe (ftail re-checks due on device); the slot-keyed dict makes
        # position swaps unable to hide a due slot behind a stale low
        # fill. Nothing-due steps are pure host (~microseconds).
        st = _mirror(pools, dev, n)
        gen = st["gen"] + 1
        cur_step = gen if step is None else int(step)
        force = bool(os.getenv("VLLM_ASCEND_GDN_LQ_FLUSH_V2_FORCE"))
        st["host_seq"][0] = gen
        # NB: Tensor.to(dst, non_blocking=True) is a SILENT NO-OP on this
        # torch_npu build - dst.copy_(src, non_blocking=True) is the
        # working async form (verified H2D and D2H).
        st["dev_seq"].copy_(st["host_seq"], non_blocking=True)
        _apply_proven_launch_copies(st, cur_step)
        # pools-side resync (init_from_dense / quantize_from_tensor)
        # recycles slots behind the mirror's back: drop every record
        epoch = getattr(pools, "_lq_epoch", 0)
        if st["epoch"] != epoch:
            st["fills"].clear()
            st["epoch"] = epoch
        launch = True
        view_g = -1
        vw = 0
        for w in (0, 1):
            g = st["buf_gen"][w]
            if g > 0 and int(st["seq_np"][w][0]) == g and g > view_g:
                view_g = g
                vw = w
        # The idx view may lag up to _VIEW_LAG decode steps: a batch
        # change is fully visible after _VIEW_LAG steps, and a newly
        # admitted slot needs 15 decode steps before it can be due, so
        # the lag cannot hide a due slot.
        reason = "noview"
        if not force and view_g > 0 and (cur_step - st["buf_step"][vw]) <= _VIEW_LAG:
            reason = "stale"
            view = st["idx_np"][vw][:st["buf_n"][vw]]
            fills = st["fills"]
            W = _WINDOW
            for raw in view:
                ent = fills.get(int(raw))
                if ent is None:
                    reason = "unknown"
                    break
                f0, s0 = ent
                if f0 >= W:
                    # fill == L records the pre-fold reading of a head
                    # that was due AND folded at observation s0. With an
                    # explicit step counter the post-fold window is
                    # sim = cur - s0 (real fill ran 1,2,...): due again
                    # when it reaches W, and a same-step double flush
                    # (cur == s0, no decode between) re-fires. Without a
                    # step counter (step=None) no credit applies at all:
                    # two consecutive fill>=L readings always re-fire,
                    # which keeps no-decode measurement loops honest.
                    if step is None or cur_step - s0 == 0 \
                            or cur_step - s0 >= W:
                        reason = "due16"
                        break
                    continue
                if f0 + (cur_step - s0) >= W:
                    reason = "due"
                    break
            else:
                launch = False
                reason = "skip"
        if os.getenv("VLLM_ASCEND_GDN_LQ_LAUNCHLOG"):
            st["reasons"] = st.get("reasons", {})
            st["reasons"][reason] = st["reasons"].get(reason, 0) + 1
            if reason not in ("skip",) and (st["reasons"][reason] % 200) == 1:
                import sys as _s
                print(f"[lq-reason] pool={id(pools)} gen={gen} step={cur_step} "
                      f"reason={reason} view_g={view_g} nfills={len(st['fills'])}",
                      file=_s.stderr, flush=True)
        if not launch:
            if gen % _VIEW_LAG == 0:
                w = gen % 2
                st["pin_idx"][w][:t].copy_(idx, non_blocking=True)
                st["pin_seq"][w].copy_(st["dev_seq"], non_blocking=True)
                st["buf_gen"][w] = gen
                st["buf_n"][w] = t
                st["buf_inst"][w] = n
                st["buf_step"][w] = cur_step
                st["buf_launch"][w] = False
            st["gen"] = gen
            return True
        flags, ab, ba, s, c, kq, vq = _bufs(t, h, dev)
        q0p = _q0_pad(pools)
        state_base = torch.as_strided(pools._state, (1,), (1,))
        slot_stride = pools._state.stride(0) * 4
        base = (int(idx.data_ptr()), int(pools._state.data_ptr()), t, h)
        ok = _fprep(state_base, idx, ab, flags, ba, h, slot_stride,
                    ("prep",) + base)
        if not ok:
            return False
        # mirror bookkeeping for the copies issued after the chain
        w = gen % 2
        st["buf_gen"][w] = gen
        st["buf_n"][w] = t
        st["buf_inst"][w] = n
        st["buf_step"][w] = cur_step
        st["buf_launch"][w] = True
        st["gen"] = gen
        st["launches"] = st.get("launches", 0) + 1
        if os.getenv("VLLM_ASCEND_GDN_LQ_LAUNCHLOG"):
            import sys as _s
            print(f"[lq-launch] pool={id(pools)} gen={gen} step={cur_step} "
                  f"t={t}", file=_s.stderr, flush=True)
        ok = _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                    h, slot_stride, 1, ("g1",) + base)
        ok = ok and _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                           h, slot_stride, 2, ("g2",) + base)
        ok = ok and _ftail(state_base, idx, s, c, vq, kq, h, slot_stride, 1,
                           ("t1",) + base)
        ok = ok and _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                           h, slot_stride, 3, ("g3",) + base)
        ok = ok and _ftail(state_base, idx, s, c, vq, kq, h, slot_stride, 2,
                           ("t2",) + base)
        ok = ok and _fgemv(state_base, idx, s, ab, flags, ba, q0p, kq, vq, c,
                           h, slot_stride, 4, ("g4",) + base)
        ok = ok and _ftail(state_base, idx, s, c, vq, kq, h, slot_stride, 3,
                           ("t3",) + base)
        st["pin"][w][:2 * n].copy_(flags.view(torch.int32).reshape(-1)[:2 * n],
                                   non_blocking=True)
        st["pin_idx"][w][:t].copy_(idx, non_blocking=True)
        st["pin_seq"][w].copy_(st["dev_seq"], non_blocking=True)
        return ok
    except Exception as e:
        import sys
        print(f"[flush-v2] exception: {type(e).__name__}: {e!r}",
              file=sys.stderr, flush=True)
        return False
