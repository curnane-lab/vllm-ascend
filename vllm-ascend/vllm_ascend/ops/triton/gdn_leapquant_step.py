# SPDX-License-Identifier: Apache-2.0
"""Triton-Ascend decode-step kernel for LeapQuant-quantized GDN state.

One program per (sequence, head). Reads the composite checkpoint (int8 codes +
fp16 Compensator Tokens) and the window's buffered rank-1 updates in place from
the layer's ssm_state slot views, computes the gated-delta-rule step, appends
the new update, and returns the token output. fp32 accumulation throughout.
"""

import torch
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
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    slot = tl.load(idx_ptr + b)

    rv = tl.arange(0, V)
    rk = tl.arange(0, K)
    rl = tl.arange(0, L)

    f = tl.load(fill_ptr + slot * fs0 + h * fhs)
    g = tl.load(g_ptr + b * H + h)
    at = tl.exp(g)
    p = tl.load(p_ptr + slot * ps0 + h * phs)
    pn = p * at
    beta = tl.load(beta_ptr + b * H + h).to(tl.float32)

    # checkpoint part: S0 = q^T u + dequant(codes), then S = pn * S0
    sv = tl.load(sv_ptr + slot * svs0 + h * svhs + rv)
    sk = tl.load(sk_ptr + slot * sks0 + h * skhs + rk)
    codes = tl.load(codes_ptr + slot * cs0 + h * chs + rv[:, None] * K + rk[None, :]).to(tl.float32)
    S = codes * (sk[None, :] / QMAX) * sv[:, None]
    for r in tl.static_range(R):
        qr = tl.load(q_ptr + slot * qs0 + h * qhs + r * V + rv).to(tl.float32)
        ur = tl.load(u_ptr + slot * us0 + h * uhs + r * K + rk).to(tl.float32)
        S += qr[:, None] * ur[None, :]
    S = pn * S

    # buffered updates: S += sum_j (w_j * at) * ubuf_j (x) kbuf_j, live entries only
    w = tl.load(w_ptr + slot * ws0 + h * whs + rl, mask=rl < f, other=0.0)
    for j in tl.static_range(L):
        ub = tl.load(ubuf_ptr + slot * ubs0 + h * ubhs + j * V + rv, mask=j < f, other=0.0).to(tl.float32)
        kb = tl.load(kbuf_ptr + slot * kbs0 + h * kbhs + j * K + rk, mask=j < f, other=0.0).to(tl.float32)
        wj = tl.load(w_ptr + slot * ws0 + h * whs + j, mask=j < f, other=0.0)
        S += (wj * at) * ub[:, None] * kb[None, :]

    # gated delta rule on the reconstructed state
    kn = tl.load(kn_ptr + b * H * K + h * K + rk)
    knf = kn.to(tl.float32)
    qn = tl.load(qn_ptr + b * H * K + h * K + rk).to(tl.float32) * scale
    val = tl.load(v_ptr + b * H * V + h * V + rv).to(tl.float32)
    Skn = tl.sum(S * knf[None, :], axis=1)
    u_new = beta * (val - Skn)
    kq = tl.sum(knf * qn, axis=0)
    o = tl.sum(S * qn[None, :], axis=1) + kq * u_new
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
    )
    return o
