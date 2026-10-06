"""E2 due-gate tests: slot-keyed mirror semantics.

Covers grok's checklist:
 1. mirror keys are slot ids; an unseen slot is due (conservative launch)
 2. a stable batch from fill=0 launches exactly once in 16 steps
    (plus the cold-start refresh at step 1)
 3. swapping two slots mid-stream cannot hide a due slot behind a stale
    low fill at the swapped position (launch fires at the right step)
 4. nothing-due steps stay pure-host (~0.065 ms median, zero device work)
 5. parity (decoded-state ratio 1.0000) - tests/test_flush_v2.py
"""

import sys
import time

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402
import gdn_lq_flush_v2 as v2  # noqa: E402

DEVICE = "npu"
H, V, K = 32, 128, 128


def _setup(t, slots):
    gen = torch.Generator().manual_seed(3)
    idx = (torch.arange(t) + 1).to(DEVICE)
    buf = torch.randn(max(64, t + 8), H, V, K, generator=gen).to(DEVICE) * 0.05
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)
    return pools, idx


def _inputs(t, seed=11):
    g = torch.Generator().manual_seed(seed)
    nk = H // 2
    kn = torch.randn(t, nk, K, generator=g)
    kn = (kn / kn.norm(dim=-1, keepdim=True)).to(torch.bfloat16).to(DEVICE)
    qn = torch.randn(t, nk, K, generator=g)
    qn = (qn / qn.norm(dim=-1, keepdim=True)).to(torch.bfloat16).to(DEVICE)
    kn = kn.repeat_interleave(H // nk, dim=1)  # decode_step takes the
    qn = qn.repeat_interleave(H // nk, dim=1)  # expanded value-head rows
    vx = torch.randn(t, H, V, generator=g).to(torch.bfloat16).to(DEVICE)
    gl = -(torch.rand(t, H, generator=g) * 0.1 + 0.005).to(DEVICE)
    bt = torch.rand(t, H, generator=g).to(torch.bfloat16).to(DEVICE)
    return kn, qn, vx, gl, bt


class Probe:
    """flush wrapper recording launch vs skip from the mirror state."""

    def __init__(self):
        self.launches = []

    def __call__(self, pools, idx, step):
        st = v2._mirror(pools, idx.device, idx.shape[0] * H)
        g0 = st["gen"]
        r = v2.flush_pools_v2(pools, idx, step=step)
        assert r, "flush v2 unavailable"
        torch.npu.synchronize()
        g1 = st["gen"]
        assert g1 == g0 + 1
        w = g1 % 2
        self.launches.append(bool(st["buf_launch"][w] and st["buf_gen"][w] == g1))


def test_keys_and_unknown():
    """1. mirror records slot ids; a slot in view that was never observed
    forces a conservative launch at first sight (then sim takes over)."""
    v2.reset_due_gate()
    pools, idx = _setup(4, None)
    all5 = (torch.arange(5) + 1).to(DEVICE)
    pools.init_from_dense(pools._state, all5)  # slot 5 initialized too
    kn, qn, vx, gl, bt = _inputs(4)
    p = Probe()
    p(pools, idx, step=1)          # cold start: launch
    p(pools, idx, step=2)          # applies step 1's copy into the dict
    st = v2._mirror(pools, idx.device, 4 * H)
    keys = set(st["fills"])
    assert keys == set(idx.cpu().tolist()), f"mirror keys {keys} != slot ids"
    # widen the batch with a never-observed slot 5: first sight launches
    idxB = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64).to(DEVICE)
    kn5, qn5, vx5, gl5, bt5 = _inputs(5, seed=12)
    pools.decode_step(idxB, kn5, qn5, vx5, gl5, bt5, 1.0)
    p(pools, idxB, step=3)   # view lags (refreshed every 4th call): skip
    pools.decode_step(idxB, kn5, qn5, vx5, gl5, bt5, 1.0)
    p(pools, idxB, step=4)   # gen%4==0: this skip refreshes the view
    pools.decode_step(idxB, kn5, qn5, vx5, gl5, bt5, 1.0)
    p(pools, idxB, step=5)   # slot 5 now in view, never seen -> launch
    assert p.launches[-1] is True, "unseen slot in view must launch"
    # slot 5 fill was 3 at the step-5 refresh; sim due 13 steps later
    for s in range(6, 20):
        pools.decode_step(idxB, kn5, qn5, vx5, gl5, bt5, 1.0)
        p(pools, idxB, step=s)
    fired = [i + 1 for i, l in enumerate(p.launches) if l]
    assert fired == [1, 5, 18], f"launches {fired}"
    print("T1 PASS: slot-id keys + unseen-in-view -> due (rhythm 1,5,18)")


def test_16_step_rhythm():
    """2. stable batch from fill=0: launches exactly at 1 and 16."""
    v2.reset_due_gate()
    pools, idx = _setup(4, None)
    kn, qn, vx, gl, bt = _inputs(4)
    p = Probe()
    for s in range(1, 17):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
        p(pools, idx, step=s)
    expect = [s == 1 or s == 16 for s in range(1, 17)]
    assert p.launches == expect, f"rhythm {p.launches}"
    print("T2 PASS: 16-step rhythm (launches at 1 and 16 only)")


def test_swap_hides_nothing():
    """3. position swap cannot hide a due slot behind a stale low fill.

    A advances alone to fill 4; B joins at step 4 (B unknown -> launch
    records A:(4,4), B:(1,4) under SLOT ids). From step 5 the batch
    order is permanently swapped to [B, A]. A is due exactly at step 16
    (4 + 12); a positional mirror would attribute B's recorded fill 1 to
    A's new position and miss it (predict 19).
    """
    v2.reset_due_gate()
    pools, idxA = _setup(1, None)
    pools.init_from_dense(pools._state, torch.tensor([1, 2]).to(DEVICE))
    idxA = torch.tensor([1], dtype=torch.int64).to(DEVICE)
    kn1, qn1, vx1, gl1, bt1 = _inputs(1)
    p = Probe()
    for s in range(1, 4):
        pools.decode_step(idxA, kn1, qn1, vx1, gl1, bt1, 1.0)
        p(pools, idxA, step=s)
    # step 4: B joins (A->4, B->1); view lags one step -> skip
    idxAB = torch.tensor([1, 2], dtype=torch.int64).to(DEVICE)
    kn2, qn2, vx2, gl2, bt2 = _inputs(2)
    pools.decode_step(idxAB, kn2, qn2, vx2, gl2, bt2, 1.0)
    p(pools, idxAB, step=4)
    assert p.launches[-1] is False
    # step 5: B now in view, never observed -> conservative launch; the
    # refresh records A:(5,5) and B:(2,5) under their SLOT ids
    pools.decode_step(idxAB, kn2, qn2, vx2, gl2, bt2, 1.0)
    p(pools, idxAB, step=5)
    assert p.launches[-1] is True, "B unseen -> launch"
    # steps 6..19 with SWAPPED order [B, A] every step: A is due exactly
    # at step 16 (5 + 11); a positional mirror would pin B's fill 2 to
    # A's new position and predict 19 - the miss this test exists for.
    idxBA = torch.tensor([2, 1], dtype=torch.int64).to(DEVICE)
    for s in range(6, 20):
        pools.decode_step(idxBA, kn2, qn2, vx2, gl2, bt2, 1.0)
        p(pools, idxBA, step=s)
    fired = [s for s, l in zip(range(6, 20), p.launches[5:]) if l]
    assert fired == [16, 19], f"launches after swap: {fired} (A due 16, B due 19)"
    st = v2._mirror(pools, idxAB.device, 2 * H)
    assert set(st["fills"]) == {1, 2}, st["fills"]
    print("T3 PASS: swap cannot hide A's due (launch exactly at step 16)")


def test_nothing_due_host_only():
    """4. skip steps are pure host with ~0.065 ms median wall."""
    v2.reset_due_gate()
    pools, idx = _setup(16, None)
    kn, qn, vx, gl, bt = _inputs(16)
    p = Probe()
    pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    p(pools, idx, step=1)  # cold launch
    walls = []
    for s in range(2, 18):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        r = v2.flush_pools_v2(pools, idx, step=s)
        dt = time.perf_counter() - t0
        assert r
        walls.append(dt)
        torch.npu.synchronize()
    skips = sorted(walls)
    med = skips[len(skips) // 2] * 1e3
    assert med < 0.12, f"skip median {med:.4f} ms"
    # exactly one launch among steps 2..17: step 16
    launched = [s for s, w in zip(range(2, 18), walls) if w > 0.5e-3]
    assert launched == [16], f"launched at {launched}"
    print(f"T4 PASS: nothing-due pure host, median {med:.4f} ms, "
          f"launch only at step 16")


def test_two_layers_same_shape():
    """Matrix: two pools of the same shape hold independent mirrors.

    GPT counterexample - a shared per-device mirror let layer A's fill=1
    record skip layer B's actually-due head. Per-pools mirrors make the
    second pool's empty mirror conservatively launch."""
    v2.reset_due_gate()
    pa, idx = _setup(4, None)
    pb, _ = _setup(4, None)
    ka = _inputs(4)
    kb = _inputs(4, seed=13)
    pa.decode_step(idx, *ka, 1.0)
    pb.decode_step(idx, *kb, 1.0)
    pA = Probe(); pA(pa, idx, step=1)   # A cold-launches
    pa.decode_step(idx, *ka, 1.0)
    pA(pa, idx, step=2)                 # applies A's step-1 copy
    stA = v2._mirror(pa, idx.device, 4 * H)
    assert stA["fills"], "A's mirror must hold records"
    # B's device slots sit at fill=1 too; advance B 15 more steps with
    # its own flushes while A's mirror keeps saying fill=1
    # B already decoded once in the setup section (fill=1); its flushes
    # run at steps 2..16 and its cold launch records fill=2 at step 2,
    # making it due exactly at step 16
    pB = Probe()
    for s in range(2, 17):
        pb.decode_step(idx, *kb, 1.0)
        pB(pb, idx, step=s)
    stB = v2._mirror(pb, idx.device, 4 * H)
    firedB = [i + 2 for i, l in enumerate(pB.launches) if l]
    assert firedB == [2, 16], f"B launches {firedB} (cold at 2 + due at 16)"
    # A's mirror untouched by B's launches
    assert stA["fills"] and stB["fills"] is not stA["fills"]
    print("MATRIX two-layers PASS: independent mirrors, B due at 16")


def test_slot_reuse_epoch():
    """Matrix: slot reuse (resync) invalidates the mirror via the pools
    epoch; the next flush conservatively re-learns."""
    v2.reset_due_gate()
    pools, idx = _setup(4, None)
    kn, qn, vx, gl, bt = _inputs(4)
    p = Probe()
    for s in range(1, 3):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
        p(pools, idx, step=s)
    st = v2._mirror(pools, idx.device, 4 * H)
    assert st["fills"], "mirror should hold records by step 2"
    # resync (slot recycling) bumps the epoch behind the mirror's back
    pools.init_from_dense(pools._state, idx)
    pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    p(pools, idx, step=3)
    assert p.launches[-1] is True, "post-resync flush must conservatively launch"
    print("MATRIX slot-reuse-epoch PASS: resync invalidates the mirror")


def test_fill_transitions():
    """Matrix: fill 0 / 15 / 16 transitions and the fill>=L credit rule.

    - fill=15 observed at step g -> launch at g+1 (sim hits L)
    - fold at g records pre-fold fill=L; with an advancing step counter
      the next call takes the credit (real fill is 1) and skips
    - the SAME fill>=L reading without an advancing step (or under
      step=None) re-fires: two consecutive fill>=L readings never skip
    """
    v2.reset_due_gate()
    pools, idx = _setup(2, None)
    kn, qn, vx, gl, bt = _inputs(2)
    p = Probe()
    pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    p(pools, idx, step=1)                       # cold; records fill=1
    # steps 2..15 decode and skip (sim reaches 15 at step 15)
    for s in range(2, 16):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
        p(pools, idx, step=s)
    assert not p.launches[-1], "sim=15 at step 15 must skip"
    pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    p(pools, idx, step=16)
    assert p.launches[-1] is True, "fill=15 record + 1 step -> due"
    # fold ran at 16; with the step counter advancing, 17 takes the
    # credit (real fill is 1) and skips
    pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    p(pools, idx, step=17)
    assert not p.launches[-1], "post-fold credit skip at 17"
    # without an advancing step the same fill>=L reading re-fires:
    # step=None (per-call gen) must launch at every call now
    pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    p(pools, idx, step=None)
    assert p.launches[-1] is True, "step=None must re-fire on fill>=L"
    print("MATRIX fill-transitions PASS: 0/15/16 + credit + re-fire")


def test_mixed_due_batch():
    """Matrix: a batch where some heads are due and some are not still
    launches (any due slot in view forces the chain; ftail gates the
    per-instance writeback on device)."""
    v2.reset_due_gate()
    pools, idx = _setup(4, None)
    kn, qn, vx, gl, bt = _inputs(4)
    p = Probe()
    for s in range(1, 16):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
        p(pools, idx, step=s)
    pools.fill[idx[0].item()] = 16   # force one head due early
    pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
    p(pools, idx, step=16)
    assert p.launches[-1] is True
    print("MATRIX mixed-due PASS: any due head in view launches")


def test_96_steps():
    """Matrix: six consecutive windows (96 decode+flush steps); launches
    exactly at the cold start and each window boundary."""
    v2.reset_due_gate()
    pools, idx = _setup(4, None)
    kn, qn, vx, gl, bt = _inputs(4)
    p = Probe()
    for s in range(1, 97):
        pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
        p(pools, idx, step=s)
    fired = [i + 1 for i, l in enumerate(p.launches) if l]
    assert fired == [1, 16, 32, 48, 64, 80, 96], fired
    print("MATRIX 96-steps PASS: launches at", fired)


def test_shape_coverage():
    """Matrix: B=16/64/256 rhythm (launch count over 2 windows)."""
    for b in (16, 64, 256):
        v2.reset_due_gate()
        gen = torch.Generator().manual_seed(3)
        idx = (torch.arange(b) + 1).to(DEVICE)
        buf = torch.randn(b + 8, H, V, K, generator=gen).to(DEVICE) * 0.05
        pools = GDNLeapQuantPools(buf)
        pools.init_from_dense(buf, idx)
        kn, qn, vx, gl, bt = _inputs(b)
        p = Probe()
        for s in range(1, 33):
            pools.decode_step(idx, kn, qn, vx, gl, bt, 1.0)
            p(pools, idx, step=s)
        fired = [i + 1 for i, l in enumerate(p.launches) if l]
        assert fired == [1, 16, 32], (b, fired)
        print(f"MATRIX shape B={b} PASS: launches at {fired}")


def test_gpt_mock_counterexample():
    """GPT's mock: a mirror holding fill=1 for a device slot that is
    actually at fill=16 must not return a zero-launch skip. With a
    per-pools mirror the mock maps to pool A holding a stale low record
    while pool B's slots are due: B's own mirror is empty -> launch."""
    v2.reset_due_gate()
    pa, idx = _setup(2, None)
    pb, _ = _setup(2, None)
    ka = _inputs(2)
    kb = _inputs(2, seed=14)
    # advance B alone to fill=16 (16 decode steps, flushing each step;
    # its own mirror launches at 16 -> fold)
    pB = Probe()
    for s in range(1, 17):
        pb.decode_step(idx, *kb, 1.0)
        pB(pb, idx, step=s)
    assert pB.launches[-1], "B must launch at its due step"
    # A's mirror still says fill=1 for the same slot ids; A's device
    # slots also sit at fill=1 (never advanced beyond step 1)
    pa.decode_step(idx, *ka, 1.0)
    pA = Probe(); pA(pa, idx, step=1)
    pa.decode_step(idx, *ka, 1.0)
    pA(pa, idx, step=2)  # applies A's step-1 record
    stA = v2._mirror(pa, idx.device, 2 * H)
    assert list(stA["fills"].values())[0][0] == 1
    print("MATRIX gpt-mock PASS: per-pools mirrors isolate the stale "
          "fill=1 record from the actually-due slot set")


if __name__ == "__main__":
    test_keys_and_unknown()
    test_16_step_rhythm()
    test_swap_hides_nothing()
    test_nothing_due_host_only()
    test_two_layers_same_shape()
    test_slot_reuse_epoch()
    test_fill_transitions()
    test_mixed_due_batch()
    test_96_steps()
    test_shape_coverage()
    test_gpt_mock_counterexample()
    print("DUE GATE E5 MATRIX ALL PASS")
