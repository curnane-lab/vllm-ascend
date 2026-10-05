# SPDX-License-Identifier: Apache-2.0
"""Triton-Ascend window-flush kernel for LeapQuant-quantized GDN state.

One program per (sequence, head); programs whose window has not ended
(fill < L) exit immediately. Due programs rebuild the fp32 state from the
composite checkpoint plus the buffered updates, fit R Compensator Tokens with
one round of subspace iteration (modified Gram-Schmidt, projections applied
twice), smooth and quantize the residual to int8, and reset the window.

The state is processed in V-chunks of VH rows so the Unified Buffer footprint
stays small: five passes (subspace start, orthogonal factor, update factor,
smoothing scales, quantization) rebuild the chunk on each pass.
"""

import torch
from vllm.triton_utils import tl, triton

QMAX: tl.constexpr = 127.0
EPS: tl.constexpr = 1e-12


@triton.jit
def _rebuild_chunk(
    codes_ptr,
    kbuf_ptr,
    ubuf_ptr,
    u_ptr,
    q_ptr,
    sk_ptr,
    sv_ptr,
    w_ptr,
    p_ptr,
    slot,
    h,
    f,
    rv_h,
    cs0,
    kbs0,
    ubs0,
    us0,
    qs0,
    sks0,
    svs0,
    ws0,
    ps0,
    V: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
):
    rk = tl.arange(0, K)
    sv_h = tl.load(sv_ptr + slot * svs0 + h * V + rv_h)
    sk_old = tl.load(sk_ptr + slot * sks0 + h * K + rk)
    codes = tl.load(codes_ptr + slot * cs0 + h * V * K + rv_h[:, None] * K + rk[None, :]).to(tl.float32)
    S = codes * (sk_old[None, :] / QMAX) * sv_h[:, None]
    for r in tl.static_range(R):
        qr = tl.load(q_ptr + slot * qs0 + (h * R + r) * V + rv_h).to(tl.float32)
        ur = tl.load(u_ptr + slot * us0 + (h * R + r) * K + rk).to(tl.float32)
        S += qr[:, None] * ur[None, :]
    p = tl.load(p_ptr + slot * ps0 + h)
    S = p * S
    for j in tl.static_range(L):
        ub = tl.load(ubuf_ptr + slot * ubs0 + (h * L + j) * V + rv_h, mask=j < f, other=0.0).to(tl.float32)
        kb = tl.load(kbuf_ptr + slot * kbs0 + (h * L + j) * K + rk, mask=j < f, other=0.0).to(tl.float32)
        wj = tl.load(w_ptr + slot * ws0 + h * L + j, mask=j < f, other=0.0)
        S += wj * ub[:, None] * kb[None, :]
    return S


@triton.jit
def _gdn_lq_flush_kernel(
    codes_ptr,
    kbuf_ptr,
    ubuf_ptr,
    u_ptr,
    q_ptr,
    sk_ptr,
    sv_ptr,
    w_ptr,
    p_ptr,
    fill_ptr,
    q0_ptr,
    idx_ptr,
    cs0,
    kbs0,
    ubs0,
    us0,
    qs0,
    sks0,
    svs0,
    ws0,
    ps0,
    fs0,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
    VH: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    slot = tl.load(idx_ptr + b)

    f = tl.load(fill_ptr + slot * fs0 + h)
    due = f >= L
    f = tl.where(due, f, L)  # clamp: non-due programs run masked, stores are guarded

    NH: tl.constexpr = V // VH
    rv = tl.arange(0, V)
    rk = tl.arange(0, K)
    rl = tl.arange(0, L)

    # pass 1: m1_r = S^T q0_r, accumulated over V-chunks; then MGS -> qk_r
    m1_0 = tl.zeros((K,), dtype=tl.float32)
    m1_1 = tl.zeros((K,), dtype=tl.float32)
    m1_2 = tl.zeros((K,), dtype=tl.float32)
    m1_3 = tl.zeros((K,), dtype=tl.float32)
    for vh in range(NH):
        rv_h = vh * VH + tl.arange(0, VH)
        S_h = _rebuild_chunk(codes_ptr, kbuf_ptr, ubuf_ptr, u_ptr, q_ptr, sk_ptr, sv_ptr,
                             w_ptr, p_ptr, slot, h, f, rv_h, cs0, kbs0, ubs0, us0, qs0,
                             sks0, svs0, ws0, ps0, V, K, L, R)
        q0_0 = tl.load(q0_ptr + 0 * V + rv_h)
        q0_1 = tl.load(q0_ptr + 1 * V + rv_h)
        q0_2 = tl.load(q0_ptr + 2 * V + rv_h)
        q0_3 = tl.load(q0_ptr + 3 * V + rv_h)
        m1_0 += tl.sum(S_h * q0_0[:, None], axis=0)
        m1_1 += tl.sum(S_h * q0_1[:, None], axis=0)
        m1_2 += tl.sum(S_h * q0_2[:, None], axis=0)
        m1_3 += tl.sum(S_h * q0_3[:, None], axis=0)
    # MGS (twice is enough), normalizing each column before the next uses it
    n0 = tl.maximum(tl.sqrt(tl.sum(m1_0 * m1_0, axis=0)), 1e-30)
    qk_0 = m1_0 / n0
    qk_1 = m1_1 - qk_0 * tl.sum(qk_0 * m1_1, axis=0)
    qk_1 = qk_1 - qk_0 * tl.sum(qk_0 * qk_1, axis=0)
    qk_1 = qk_1 / tl.maximum(tl.sqrt(tl.sum(qk_1 * qk_1, axis=0)), 1e-30)
    qk_2 = m1_2 - qk_0 * tl.sum(qk_0 * m1_2, axis=0)
    qk_2 = qk_2 - qk_1 * tl.sum(qk_1 * qk_2, axis=0)
    qk_2 = qk_2 - qk_0 * tl.sum(qk_0 * qk_2, axis=0)
    qk_2 = qk_2 - qk_1 * tl.sum(qk_1 * qk_2, axis=0)
    qk_2 = qk_2 / tl.maximum(tl.sqrt(tl.sum(qk_2 * qk_2, axis=0)), 1e-30)
    qk_3 = m1_3 - qk_0 * tl.sum(qk_0 * m1_3, axis=0)
    qk_3 = qk_3 - qk_1 * tl.sum(qk_1 * qk_3, axis=0)
    qk_3 = qk_3 - qk_2 * tl.sum(qk_2 * qk_3, axis=0)
    qk_3 = qk_3 - qk_0 * tl.sum(qk_0 * qk_3, axis=0)
    qk_3 = qk_3 - qk_1 * tl.sum(qk_1 * qk_3, axis=0)
    qk_3 = qk_3 - qk_2 * tl.sum(qk_2 * qk_3, axis=0)
    qk_3 = qk_3 / tl.maximum(tl.sqrt(tl.sum(qk_3 * qk_3, axis=0)), 1e-30)

    # pass 2: qv_r = S qk_r over V-chunks; then MGS -> qv_r
    qv_0 = tl.zeros((V,), dtype=tl.float32)
    qv_1 = tl.zeros((V,), dtype=tl.float32)
    qv_2 = tl.zeros((V,), dtype=tl.float32)
    qv_3 = tl.zeros((V,), dtype=tl.float32)
    for vh in range(NH):
        rv_h = vh * VH + tl.arange(0, VH)
        S_h = _rebuild_chunk(codes_ptr, kbuf_ptr, ubuf_ptr, u_ptr, q_ptr, sk_ptr, sv_ptr,
                             w_ptr, p_ptr, slot, h, f, rv_h, cs0, kbs0, ubs0, us0, qs0,
                             sks0, svs0, ws0, ps0, V, K, L, R)
        OH = rv[:, None] == rv_h[None, :]
        qv_0 += tl.sum(tl.where(OH, tl.sum(S_h * qk_0[None, :], axis=1)[None, :], 0.0), axis=1)
        qv_1 += tl.sum(tl.where(OH, tl.sum(S_h * qk_1[None, :], axis=1)[None, :], 0.0), axis=1)
        qv_2 += tl.sum(tl.where(OH, tl.sum(S_h * qk_2[None, :], axis=1)[None, :], 0.0), axis=1)
        qv_3 += tl.sum(tl.where(OH, tl.sum(S_h * qk_3[None, :], axis=1)[None, :], 0.0), axis=1)
    n0 = tl.maximum(tl.sqrt(tl.sum(qv_0 * qv_0, axis=0)), 1e-30)
    qv_0 = qv_0 / n0
    qv_1 = qv_1 - qv_0 * tl.sum(qv_0 * qv_1, axis=0)
    qv_1 = qv_1 - qv_0 * tl.sum(qv_0 * qv_1, axis=0)
    qv_1 = qv_1 / tl.maximum(tl.sqrt(tl.sum(qv_1 * qv_1, axis=0)), 1e-30)
    qv_2 = qv_2 - qv_0 * tl.sum(qv_0 * qv_2, axis=0)
    qv_2 = qv_2 - qv_1 * tl.sum(qv_1 * qv_2, axis=0)
    qv_2 = qv_2 - qv_0 * tl.sum(qv_0 * qv_2, axis=0)
    qv_2 = qv_2 - qv_1 * tl.sum(qv_1 * qv_2, axis=0)
    qv_2 = qv_2 / tl.maximum(tl.sqrt(tl.sum(qv_2 * qv_2, axis=0)), 1e-30)
    qv_3 = qv_3 - qv_0 * tl.sum(qv_0 * qv_3, axis=0)
    qv_3 = qv_3 - qv_1 * tl.sum(qv_1 * qv_3, axis=0)
    qv_3 = qv_3 - qv_2 * tl.sum(qv_2 * qv_3, axis=0)
    qv_3 = qv_3 - qv_0 * tl.sum(qv_0 * qv_3, axis=0)
    qv_3 = qv_3 - qv_1 * tl.sum(qv_1 * qv_3, axis=0)
    qv_3 = qv_3 - qv_2 * tl.sum(qv_2 * qv_3, axis=0)
    qv_3 = qv_3 / tl.maximum(tl.sqrt(tl.sum(qv_3 * qv_3, axis=0)), 1e-30)

    # pass 3: uk_r = S^T qv_r over V-chunks
    uk_0 = tl.zeros((K,), dtype=tl.float32)
    uk_1 = tl.zeros((K,), dtype=tl.float32)
    uk_2 = tl.zeros((K,), dtype=tl.float32)
    uk_3 = tl.zeros((K,), dtype=tl.float32)
    for vh in range(NH):
        rv_h = vh * VH + tl.arange(0, VH)
        S_h = _rebuild_chunk(codes_ptr, kbuf_ptr, ubuf_ptr, u_ptr, q_ptr, sk_ptr, sv_ptr,
                             w_ptr, p_ptr, slot, h, f, rv_h, cs0, kbs0, ubs0, us0, qs0,
                             sks0, svs0, ws0, ps0, V, K, L, R)
        OH = rv[:, None] == rv_h[None, :]
        qv_0h = tl.sum(tl.where(OH, qv_0[:, None], 0.0), axis=0)
        qv_1h = tl.sum(tl.where(OH, qv_1[:, None], 0.0), axis=0)
        qv_2h = tl.sum(tl.where(OH, qv_2[:, None], 0.0), axis=0)
        qv_3h = tl.sum(tl.where(OH, qv_3[:, None], 0.0), axis=0)
        uk_0 += tl.sum(S_h * qv_0h[:, None], axis=0)
        uk_1 += tl.sum(S_h * qv_1h[:, None], axis=0)
        uk_2 += tl.sum(S_h * qv_2h[:, None], axis=0)
        uk_3 += tl.sum(S_h * qv_3h[:, None], axis=0)

    # pass 4a: column absolute sums across all V-chunks (sk needs the full mean)
    sum_abs = tl.zeros((K,), dtype=tl.float32)
    for vh in range(NH):
        rv_h = vh * VH + tl.arange(0, VH)
        S_h = _rebuild_chunk(codes_ptr, kbuf_ptr, ubuf_ptr, u_ptr, q_ptr, sk_ptr, sv_ptr,
                             w_ptr, p_ptr, slot, h, f, rv_h, cs0, kbs0, ubs0, us0, qs0,
                             sks0, svs0, ws0, ps0, V, K, L, R)
        OH = rv[:, None] == rv_h[None, :]
        qv_0h = tl.sum(tl.where(OH, qv_0[:, None], 0.0), axis=0)
        qv_1h = tl.sum(tl.where(OH, qv_1[:, None], 0.0), axis=0)
        qv_2h = tl.sum(tl.where(OH, qv_2[:, None], 0.0), axis=0)
        qv_3h = tl.sum(tl.where(OH, qv_3[:, None], 0.0), axis=0)
        E = S_h - qv_0h[:, None] * uk_0[None, :] - qv_1h[:, None] * uk_1[None, :] - qv_2h[:, None] * uk_2[None, :] - qv_3h[:, None] * uk_3[None, :]
        sum_abs += tl.sum(tl.abs(E), axis=0)
    sk = tl.maximum(tl.sqrt(sum_abs / V), EPS)
    tl.store(sk_ptr + slot * sks0 + h * K + rk, sk, mask=due & (rk >= 0))

    # pass 4b: row scales (max over k of the smoothed residual) per chunk
    sv_new = tl.zeros((V,), dtype=tl.float32)
    for vh in range(NH):
        rv_h = vh * VH + tl.arange(0, VH)
        S_h = _rebuild_chunk(codes_ptr, kbuf_ptr, ubuf_ptr, u_ptr, q_ptr, sk_ptr, sv_ptr,
                             w_ptr, p_ptr, slot, h, f, rv_h, cs0, kbs0, ubs0, us0, qs0,
                             sks0, svs0, ws0, ps0, V, K, L, R)
        OH = rv[:, None] == rv_h[None, :]
        qv_0h = tl.sum(tl.where(OH, qv_0[:, None], 0.0), axis=0)
        qv_1h = tl.sum(tl.where(OH, qv_1[:, None], 0.0), axis=0)
        qv_2h = tl.sum(tl.where(OH, qv_2[:, None], 0.0), axis=0)
        qv_3h = tl.sum(tl.where(OH, qv_3[:, None], 0.0), axis=0)
        E = S_h - qv_0h[:, None] * uk_0[None, :] - qv_1h[:, None] * uk_1[None, :] - qv_2h[:, None] * uk_2[None, :] - qv_3h[:, None] * uk_3[None, :]
        sv_h = tl.max(tl.abs(E) / sk[None, :], axis=1)
        sv_new = tl.maximum(sv_new, tl.sum(tl.where(OH, sv_h[None, :], 0.0), axis=1))
    sv_new = tl.maximum(sv_new, EPS)

    # pass 5: quantize the residual per chunk
    for vh in range(NH):
        rv_h = vh * VH + tl.arange(0, VH)
        S_h = _rebuild_chunk(codes_ptr, kbuf_ptr, ubuf_ptr, u_ptr, q_ptr, sk_ptr, sv_ptr,
                             w_ptr, p_ptr, slot, h, f, rv_h, cs0, kbs0, ubs0, us0, qs0,
                             sks0, svs0, ws0, ps0, V, K, L, R)
        OH = rv[:, None] == rv_h[None, :]
        qv_0h = tl.sum(tl.where(OH, qv_0[:, None], 0.0), axis=0)
        qv_1h = tl.sum(tl.where(OH, qv_1[:, None], 0.0), axis=0)
        qv_2h = tl.sum(tl.where(OH, qv_2[:, None], 0.0), axis=0)
        qv_3h = tl.sum(tl.where(OH, qv_3[:, None], 0.0), axis=0)
        E = S_h - qv_0h[:, None] * uk_0[None, :] - qv_1h[:, None] * uk_1[None, :] - qv_2h[:, None] * uk_2[None, :] - qv_3h[:, None] * uk_3[None, :]
        sv_h = tl.sum(tl.where(OH, sv_new[:, None], 0.0), axis=0)
        scaled = E / (sk[None, :] * sv_h[:, None]) * QMAX
        qcodes = tl.where(scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
        qcodes = tl.minimum(tl.maximum(qcodes, -QMAX), QMAX)
        tl.store(codes_ptr + slot * cs0 + h * V * K + rv_h[:, None] * K + rk[None, :], qcodes.to(tl.int8), mask=due & (rv_h[:, None] >= 0))

    tl.store(sv_ptr + slot * svs0 + h * V + rv, sv_new, mask=due & (rv >= 0))
    tl.store(u_ptr + slot * us0 + (h * R + 0) * K + rk, uk_0.to(tl.float16), mask=due & (rk >= 0))
    tl.store(u_ptr + slot * us0 + (h * R + 1) * K + rk, uk_1.to(tl.float16), mask=due & (rk >= 0))
    tl.store(u_ptr + slot * us0 + (h * R + 2) * K + rk, uk_2.to(tl.float16), mask=due & (rk >= 0))
    tl.store(u_ptr + slot * us0 + (h * R + 3) * K + rk, uk_3.to(tl.float16), mask=due & (rk >= 0))
    tl.store(q_ptr + slot * qs0 + (h * R + 0) * V + rv, qv_0.to(tl.float16), mask=due & (rv >= 0))
    tl.store(q_ptr + slot * qs0 + (h * R + 1) * V + rv, qv_1.to(tl.float16), mask=due & (rv >= 0))
    tl.store(q_ptr + slot * qs0 + (h * R + 2) * V + rv, qv_2.to(tl.float16), mask=due & (rv >= 0))
    tl.store(q_ptr + slot * qs0 + (h * R + 3) * V + rv, qv_3.to(tl.float16), mask=due & (rv >= 0))

    zero_w = tl.zeros((L,), dtype=tl.float32)
    tl.store(w_ptr + slot * ws0 + h * L + rl, zero_w, mask=due & (rl >= 0))
    tl.store(p_ptr + slot * ps0 + h, 1.0, mask=due)
    tl.store(fill_ptr + slot * fs0 + h, 0, mask=due)


def gdn_leapquant_flush(pools, idx):
    """Flush due slots (fill == L) of the decode batch idx in place."""
    T = idx.shape[0]
    H = pools.num_heads
    grid = (T * H,)
    _gdn_lq_flush_kernel[grid](
        pools.codes,
        pools.kbuf,
        pools.ubuf,
        pools.u,
        pools.q,
        pools.s_k,
        pools.s_v,
        pools.w,
        pools.p,
        pools.fill,
        pools.q0,
        idx,
        pools.codes.stride(0),
        pools.kbuf.stride(0),
        pools.ubuf.stride(0),
        pools.u.stride(0),
        pools.q.stride(0),
        pools.s_k.stride(0),
        pools.s_v.stride(0),
        pools.w.stride(0),
        pools.p.stride(0),
        pools.fill.stride(0),
        H=H,
        V=pools.head_v,
        K=pools.head_k,
        L=pools.kbuf.shape[2],
        R=pools.u.shape[2],
        VH=32,
    )
