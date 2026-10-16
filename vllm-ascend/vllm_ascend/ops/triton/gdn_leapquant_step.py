# SPDX-License-Identifier: Apache-2.0
"""Triton-Ascend decode-step kernel for LeapQuant-quantized GDN state.

One program per (sequence, head). Reads the composite checkpoint (int8 codes +
fp16 Compensator Tokens) and the window's buffered rank-1 updates in place from
the layer's ssm_state slot views, computes the gated-delta-rule step, appends
the new update, and returns the token output. fp32 accumulation throughout.
"""

import torch
import os
from vllm.triton_utils import tl, triton

QMAX: tl.constexpr = 127.0


@triton.jit
def _gdn_lq_step_kernel(
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
    kn_ptr,
    qn_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    o_ptr,
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
    chs,
    kbhs,
    ubhs,
    uhs,
    qhs,
    skhs,
    svhs,
    whs,
    phs,
    fhs,
    scale,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    R: tl.constexpr,
    DIRECT_READOUT: tl.constexpr,
    VBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    slot = tl.load(idx_ptr + b)

    rv = tl.arange(0, V)
    rk = tl.arange(0, K)
    rl = tl.arange(0, L)
    rvb = tl.arange(0, VBLOCK)

    f = tl.load(fill_ptr + slot * fs0 + h * fhs)
    # fill == L: window full and flush already folded it - this step opens
    # the new window at entry 0 (paper semantics)
    f = tl.where(f >= L, 0, f)
    g = tl.load(g_ptr + b * H + h)
    at = tl.exp(g)
    p = tl.load(p_ptr + slot * ps0 + h * phs)
    pn = p * at
    beta = tl.load(beta_ptr + b * H + h).to(tl.float32)

    kn = tl.load(kn_ptr + b * H * K + h * K + rk)
    knf = kn.to(tl.float32)
    qn = tl.load(qn_ptr + b * H * K + h * K + rk).to(tl.float32) * scale
    val = tl.load(v_ptr + b * H * V + h * V + rv).to(tl.float32)

    if DIRECT_READOUT:
        # Direct readout: never materialize S. Compute Sx for key/query from
        # dequant rows, compensators, and live window entries. This trades
        # V*K outer-product accumulation for K-length dot products.
        sk = tl.load(sk_ptr + slot * sks0 + h * skhs + rk)

        sk_scaled_kn = sk / QMAX * knf
        sk_scaled_qn = sk / QMAX * qn
        checkpoint_kn = tl.zeros((V,), dtype=tl.float32)
        checkpoint_qn = tl.zeros((V,), dtype=tl.float32)
        for v0 in range(0, V, VBLOCK):
            vmask = rvb < V - v0
            code_rows = tl.load(
                codes_ptr + slot * cs0 + h * chs + (v0 + rvb)[:, None] * K + rk[None, :],
                mask=vmask[:, None],
                other=0.0,
            ).to(tl.float32)
            sv_rows = tl.load(
                sv_ptr + slot * svs0 + h * svhs + v0 + rvb,
                mask=vmask,
                other=0.0,
            )
            block_kn = tl.sum(
                code_rows * (sv_rows / QMAX)[:, None] * sk_scaled_kn[None, :],
                axis=1,
            )
            block_qn = tl.sum(
                code_rows * (sv_rows / QMAX)[:, None] * sk_scaled_qn[None, :],
                axis=1,
            )
            checkpoint_kn += block_kn
            checkpoint_qn += block_qn
        checkpoint_kn *= pn
        checkpoint_qn *= pn

        window_kn = tl.zeros((V,), dtype=tl.float32)
        window_qn = tl.zeros((V,), dtype=tl.float32)
        for j in tl.static_range(L):
            if j < f:
                ub = tl.load(ubuf_ptr + slot * ubs0 + h * ubhs + j * V + rv).to(tl.float32)
                kb = tl.load(kbuf_ptr + slot * kbs0 + h * kbhs + j * K + rk).to(tl.float32)
                wj = tl.load(w_ptr + slot * ws0 + h * whs + j)
                weight = wj * at
                kb_dot_kn = tl.sum(kb * knf, axis=0)
                kb_dot_qn = tl.sum(kb * qn, axis=0)
                window_kn += weight * kb_dot_kn * ub
                window_qn += weight * kb_dot_qn * ub

        Skn = checkpoint_kn + window_kn
        state_dot_qn = checkpoint_qn + window_qn
    else:
        sv = tl.load(sv_ptr + slot * svs0 + h * svhs + rv)
        sk = tl.load(sk_ptr + slot * sks0 + h * skhs + rk)
        codes = tl.load(codes_ptr + slot * cs0 + h * chs + rv[:, None] * K + rk[None, :]).to(tl.float32)
        S = codes * (sk[None, :] / QMAX) * sv[:, None]
        for r in tl.static_range(R):
            qr = tl.load(q_ptr + slot * qs0 + h * qhs + r * V + rv).to(tl.float32)
            ur = tl.load(u_ptr + slot * us0 + h * uhs + r * K + rk).to(tl.float32)
            S += qr[:, None] * ur[None, :]
        S = pn * S

        w = tl.load(w_ptr + slot * ws0 + h * whs + rl, mask=rl < f, other=0.0)
        for j in tl.static_range(L):
            ub = tl.load(ubuf_ptr + slot * ubs0 + h * ubhs + j * V + rv, mask=j < f, other=0.0).to(tl.float32)
            kb = tl.load(kbuf_ptr + slot * kbs0 + h * kbhs + j * K + rk, mask=j < f, other=0.0).to(tl.float32)
            wj = tl.load(w_ptr + slot * ws0 + h * whs + j, mask=j < f, other=0.0)
            S += (wj * at) * ub[:, None] * kb[None, :]

        Skn = tl.sum(S * knf[None, :], axis=1)
        state_dot_qn = tl.sum(S * qn[None, :], axis=1)

    # gated delta rule on the reconstructed state
    u_new = beta * (val - Skn)
    kq = tl.sum(knf * qn, axis=0)
    o = state_dot_qn + kq * u_new
    tl.store(o_ptr + b * H * V + h * V + rv, o.to(o_ptr.dtype.element_ty))

    # append the new update at entry f
    tl.store(kbuf_ptr + slot * kbs0 + h * kbhs + f * K + rk, kn)
    tl.store(ubuf_ptr + slot * ubs0 + h * ubhs + f * V + rv, u_new.to(ubuf_ptr.dtype.element_ty))
    wnew = tl.where(rl < f, w * at, tl.where(rl == f, 1.0, 0.0))
    tl.store(w_ptr + slot * ws0 + h * whs + rl, wnew)
    tl.store(p_ptr + slot * ps0 + h * phs, pn)
    tl.store(fill_ptr + slot * fs0 + h * fhs, f + 1)


def gdn_leapquant_step(pools, idx, kn, qn, v, g_log, beta, scale):
    """LeapQuant GDN decode step on the composite pools (see gdn_leapquant.py).

    kn, qn: [T, H, K] bf16 (expanded to value heads, qn unscaled); v: [T, H, V]
    bf16; g_log: [T, H] fp32; beta: [T, H] bf16; idx: [T] int64 slot ids.
    Appends the update in place and returns o [T, H, V] bf16.
    """
    T = idx.shape[0]
    H, V, K = pools.num_heads, pools.head_v, pools.head_k
    L = pools.kbuf.shape[2]
    R = pools.u.shape[2]
    o = torch.empty(T, H, V, dtype=torch.bfloat16, device=idx.device)
    grid = (T * H,)
    direct = os.getenv("VLLM_ASCEND_GDN_LQ_DIRECT_READOUT", "0") == "1"
    vblock = int(os.getenv("VLLM_ASCEND_GDN_LQ_DIRECT_VBLOCK", "16"))
    _gdn_lq_step_kernel[grid](
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
        kn,
        qn,
        v,
        g_log.contiguous(),
        beta.contiguous(),
        o,
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
        pools.codes.stride(1),
        pools.kbuf.stride(1),
        pools.ubuf.stride(1),
        pools.u.stride(1),
        pools.q.stride(1),
        pools.s_k.stride(1),
        pools.s_v.stride(1),
        pools.w.stride(1),
        pools.p.stride(1),
        pools.fill.stride(1),
        scale,
        H=H,
        V=V,
        K=K,
        L=L,
        R=R,
        DIRECT_READOUT=direct,
        VBLOCK=vblock,
    )
    return o
