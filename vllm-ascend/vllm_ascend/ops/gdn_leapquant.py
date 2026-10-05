# SPDX-License-Identifier: Apache-2.0
"""LeapQuant per-window recurrent-state quantization for GDN decode (torch prototype).

Port of the LeapQuant GDN contracts (arXiv:2609.38166, NVlabs/kda PR #10):
the recurrent state is kept as an int8 smoothed residual plus a few fp16
Compensator Tokens, the last WINDOW rank-1 updates are buffered in bf16, and
the state is re-quantized once per window by ``flush``.

Storage is in-place: the composite checkpoint lives inside the layer's own
fp32 ssm_state slot buffer (a slot is 2 MB, the composite layout needs
~0.9 MB), so the prototype adds no extra device memory. A slot holds either
the dense fp32 state (immediately after a dense-path write, before resync)
or the composite format (every other time in LeapQuant mode).

This module is the eager torch path used to validate end-to-end accuracy on
real weights before the Triton/AscendC kernels replace the hot spots.
"""

import torch

QMAX = 127.0
EPS = 1e-12
WINDOW = 16
NUM_COMP_TOKENS = 4
_MGS_PASSES = 2
_Q0_SEED = 20260930

# byte offsets within one slot buffer (all 128-byte aligned)
_OFF_CODES = 0
_OFF_KBUF = 524288  # 32*128*128 int8
_OFF_UBUF = 655360  # + 32*16*128*2
_OFF_U = 786432  # + 32*16*128*2
_OFF_Q = 819200  # + 32*4*128*2
_OFF_SK = 852992  # + 32*4*128*2
_OFF_SV = 869376  # + 32*128*4
_OFF_W = 885760  # + 32*128*4
_OFF_P = 887808  # + 32*16*4
_OFF_FILL = 887936  # + 32*4
_LAYOUT_END = 888068  # + 32*4


def _mgs(m: torch.Tensor) -> torch.Tensor:
    """Modified Gram-Schmidt on the columns of m, projections applied twice."""
    cols = []
    for j in range(m.shape[-1]):
        v = m[..., j]
        for _ in range(_MGS_PASSES):
            for c in cols:
                v = v - c * (c * v).sum(-1, keepdim=True)
        cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    return torch.stack(cols, -1)


def _quantize(S: torch.Tensor, q0: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Quantize a dense fp32 state [..., V, K] into a LeapQuant checkpoint."""
    x = S.transpose(-1, -2)  # [..., K, V]
    qk = _mgs(x @ q0)  # [..., K, R]
    qv = _mgs(x.transpose(-1, -2) @ qk)  # [..., V, R]
    uk = x @ qv  # [..., K, R]
    u_new = uk.transpose(-1, -2).to(torch.float16)  # [..., R, K]
    q_new = qv.transpose(-1, -2).to(torch.float16)  # [..., R, V]
    e = S - torch.einsum("...rv,...rk->...vk", q_new.float(), u_new.float())
    s_k = e.abs().mean(dim=-2).sqrt().clamp_min(EPS)
    s_v = (e.abs() / s_k.unsqueeze(-2)).amax(dim=-1).clamp_min(EPS)
    codes = torch.round(e / (s_k.unsqueeze(-2) * s_v.unsqueeze(-1)) * QMAX)
    codes = codes.clamp(-QMAX, QMAX).to(torch.int8)
    return codes, s_k, s_v, u_new, q_new


class GDNLeapQuantPools:
    """In-place composite-state views over the layer's ssm_state buffer.

    ``ssm_state`` is [num_slots, H, V, K] fp32 (possibly a strided view into a
    unified KV buffer). The first ``_LAYOUT_END`` bytes of each slot's storage
    hold the LeapQuant layout; the rest is unused.
    """

    def __init__(self, ssm_state: torch.Tensor):
        assert ssm_state.dtype == torch.float32, f"expect fp32 state, got {ssm_state.dtype}"
        self.num_slots, self.num_heads, self.head_v, self.head_k = ssm_state.shape
        H, V, K = self.num_heads, self.head_v, self.head_k
        self.slot_stride_bytes = ssm_state.stride(0) * 4
        assert self.slot_stride_bytes >= _LAYOUT_END, "slot buffer too small for LeapQuant layout"
        i8 = ssm_state.view(torch.int8)  # [num_slots, H, V, K*4]; last-dim fp32->int8 view
        slot_bytes = self.num_heads * self.head_v * self.head_k * 4
        flat = torch.as_strided(i8, (self.num_slots, slot_bytes), (i8.stride(0), 1))
        self._flat = flat

        def typed(offset: int, dtype: torch.dtype, *tail: int) -> torch.Tensor:
            n = 1
            for d in tail:
                n *= d
            esize = torch.tensor([], dtype=dtype).element_size()
            cols = flat[:, offset : offset + n * esize]
            return cols.view(dtype).view(self.num_slots, *tail)

        self.codes = typed(_OFF_CODES, torch.int8, H, V, K)
        self.kbuf = typed(_OFF_KBUF, torch.bfloat16, H, WINDOW, K)
        self.ubuf = typed(_OFF_UBUF, torch.bfloat16, H, WINDOW, V)
        self.u = typed(_OFF_U, torch.float16, H, NUM_COMP_TOKENS, K)
        self.q = typed(_OFF_Q, torch.float16, H, NUM_COMP_TOKENS, V)
        self.s_k = typed(_OFF_SK, torch.float32, H, K)
        self.s_v = typed(_OFF_SV, torch.float32, H, V)
        self.w = typed(_OFF_W, torch.float32, H, WINDOW)
        self.p = typed(_OFF_P, torch.float32, H)
        self.fill = typed(_OFF_FILL, torch.int32, H)
        gen = torch.Generator(device="cpu").manual_seed(_Q0_SEED)
        self.q0 = torch.randn(V, NUM_COMP_TOKENS, dtype=torch.float32, generator=gen).to(ssm_state.device)

    def th_index(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = idx.shape[0]
        t_idx = idx.view(T, 1).expand(T, self.num_heads)
        h_idx = torch.arange(self.num_heads, device=idx.device).view(1, -1).expand(T, self.num_heads)
        return t_idx, h_idx

    def decode_step(
        self,
        idx: torch.Tensor,
        kn: torch.Tensor,
        qn: torch.Tensor,
        v: torch.Tensor,
        g_log: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """One decode step over a batch of sequences.

        kn, qn: [T, H, K] l2-normalized bf16 (already expanded to value heads,
            qn without the K**-0.5 factor); v: [T, H, V] bf16;
            g_log: [T, H] fp32 log-decay; beta: [T, H] bf16; idx: [T] slot ids.
        Returns o [T, H, V] bf16 and appends the rank-1 update in place.
        """
        t_idx, h_idx = self.th_index(idx)
        L = WINDOW

        codes = self.codes[t_idx, h_idx].float()
        s_k = self.s_k[t_idx, h_idx]
        s_v = self.s_v[t_idx, h_idx]
        s0 = torch.einsum("bhrv,bhrk->bhvk", self.q[t_idx, h_idx].float(), self.u[t_idx, h_idx].float())
        s0 = s0 + codes * (s_k / QMAX).unsqueeze(-2) * s_v.unsqueeze(-1)

        p = self.p[t_idx, h_idx]
        at = torch.exp(g_log.float())
        pn = p * at
        f = self.fill[t_idx, h_idx]
        w = self.w[t_idx, h_idx] * at.unsqueeze(-1)  # dead entries stay 0
        # dead buffer slots may hold stale bytes that are not finite as bf16;
        # mask them out so 0 * NaN does not poison the reconstruction
        j = torch.arange(L, device=idx.device)
        live = (j.view(1, 1, L) < f.unsqueeze(-1)).unsqueeze(-1)
        ubuf = torch.where(live, self.ubuf[t_idx, h_idx].float(), torch.zeros((), device=idx.device))
        kbuf = torch.where(live, self.kbuf[t_idx, h_idx].float(), torch.zeros((), device=idx.device))
        s = pn.unsqueeze(-1).unsqueeze(-1) * s0 + torch.einsum("bhj,bhjv,bhjk->bhvk", w, ubuf, kbuf)

        beta_f = beta.float()
        upd = beta_f.unsqueeze(-1) * (v.float() - torch.einsum("bhvk,bhk->bhv", s, kn.float()))
        qn_s = qn.float() * scale
        o = torch.einsum("bhvk,bhk->bhv", s, qn_s) + (kn.float() * qn_s).sum(-1, keepdim=True) * upd

        # append the new update at entry f
        self.kbuf[t_idx, h_idx, f] = kn.to(torch.bfloat16)
        self.ubuf[t_idx, h_idx, f] = upd.to(torch.bfloat16)
        j = torch.arange(L, device=idx.device)
        w_new = torch.where(j.view(1, 1, L) == f.unsqueeze(-1), torch.ones_like(w), w)
        self.w[t_idx, h_idx] = w_new
        self.p[t_idx, h_idx] = pn
        self.fill[t_idx, h_idx] = f + 1
        return o.to(torch.bfloat16)

    @torch.no_grad()
    def flush(self, idx: torch.Tensor) -> None:
        """Re-quantize the slots of idx whose window just ended (fill == L)."""
        t_idx, h_idx = self.th_index(idx)
        f = self.fill[t_idx, h_idx]
        due4 = (f >= WINDOW).unsqueeze(-1).unsqueeze(-1)
        due3 = (f >= WINDOW).unsqueeze(-1)
        due2 = f >= WINDOW

        codes = self.codes[t_idx, h_idx].float()
        s_k = self.s_k[t_idx, h_idx]
        s_v = self.s_v[t_idx, h_idx]
        s_old = torch.einsum("bhrv,bhrk->bhvk", self.q[t_idx, h_idx].float(), self.u[t_idx, h_idx].float())
        s_old = s_old + codes * (s_k / QMAX).unsqueeze(-2) * s_v.unsqueeze(-1)
        # mask dead buffer slots: stale bytes may be non-finite as bf16
        j = torch.arange(WINDOW, device=idx.device)
        live = (j.view(1, 1, WINDOW) < f.unsqueeze(-1)).unsqueeze(-1)
        ubuf = torch.where(live, self.ubuf[t_idx, h_idx].float(), torch.zeros((), device=idx.device))
        kbuf = torch.where(live, self.kbuf[t_idx, h_idx].float(), torch.zeros((), device=idx.device))
        s = self.p[t_idx, h_idx].unsqueeze(-1).unsqueeze(-1) * s_old + torch.einsum(
            "bhj,bhjv,bhjk->bhvk",
            self.w[t_idx, h_idx],
            ubuf,
            kbuf,
        )

        c, sk, sv, u_new, q_new = _quantize(s, self.q0)
        self.codes[t_idx, h_idx] = torch.where(due4, c, self.codes[t_idx, h_idx])
        self.s_k[t_idx, h_idx] = torch.where(due3, sk, self.s_k[t_idx, h_idx])
        self.s_v[t_idx, h_idx] = torch.where(due3, sv, self.s_v[t_idx, h_idx])
        self.u[t_idx, h_idx] = torch.where(due4, u_new, self.u[t_idx, h_idx])
        self.q[t_idx, h_idx] = torch.where(due4, q_new, self.q[t_idx, h_idx])
        self.w[t_idx, h_idx] = torch.where(due3, torch.zeros_like(self.w[t_idx, h_idx]), self.w[t_idx, h_idx])
        self.p[t_idx, h_idx] = torch.where(due2, torch.ones_like(self.p[t_idx, h_idx]), self.p[t_idx, h_idx])
        self.fill[t_idx, h_idx] = torch.where(due2, torch.zeros_like(f), f)

    @torch.no_grad()
    def init_from_dense(self, ssm_state: torch.Tensor, idx: torch.Tensor) -> None:
        """Quantize the dense fp32 states of slots idx into composite checkpoints.

        Must be called right after a dense-path write to those slots; reads the
        dense values via a gather first, so in-place aliasing is safe.
        """
        t_idx, h_idx = self.th_index(idx)
        dense = ssm_state[idx].clone().float()
        c, sk, sv, u_new, q_new = _quantize(dense, self.q0)
        self.codes[t_idx, h_idx] = c
        self.s_k[t_idx, h_idx] = sk
        self.s_v[t_idx, h_idx] = sv
        self.u[t_idx, h_idx] = u_new
        self.q[t_idx, h_idx] = q_new
        self.w[t_idx, h_idx] = 0.0
        self.p[t_idx, h_idx] = 1.0
        self.fill[t_idx, h_idx] = 0
        # the stale dense bytes behind the buffer views are not finite as bf16
        self.kbuf[t_idx, h_idx] = 0.0
        self.ubuf[t_idx, h_idx] = 0.0

    def decode_state(self, idx: torch.Tensor) -> torch.Tensor:
        """Dense fp32 reconstruction of the checkpoint part (window buffer excluded)."""
        t_idx, h_idx = self.th_index(idx)
        s = torch.einsum("bhrv,bhrk->bhvk", self.q[t_idx, h_idx].float(), self.u[t_idx, h_idx].float())
        return s + self.codes[t_idx, h_idx].float() * (self.s_k[t_idx, h_idx] / QMAX).unsqueeze(
            -2
        ) * self.s_v[t_idx, h_idx].unsqueeze(-1)
