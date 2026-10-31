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
_Q0_SEED = 20260930

# Per-(slot, head) contiguous block layout. Each head owns one HEAD_BYTES
# block inside its slot's storage: the ~19 KB checkpoint first (codes, scales,
# compensator tokens), then the bf16 window buffer, counters last. One DMA
# fetches a whole head — aligned with the NVlabs/kda PR #10 pool geometry.
HEAD_BYTES = 32768

# byte offsets within one head block
_OFF_CODES = 0  # int8 [V, K] = 16384
_OFF_SK = 16384  # fp32 [K] = 512
_OFF_SV = 16896  # fp32 [V] = 512
_OFF_U = 17408  # fp16 [R, K] = 1024
_OFF_Q = 18432  # fp16 [R, V] = 1024
_OFF_KBUF = 19456  # bf16 [L, K] = 4096
_OFF_UBUF = 23552  # bf16 [L, V] = 4096
_OFF_W = 27648  # fp32 [L] = 64
_OFF_P = 27712  # fp32 = 4
_OFF_FILL = 27716  # int32 = 4
_LAYOUT_END = 27720


def _mgs(m: torch.Tensor) -> torch.Tensor:
    """Modified Gram-Schmidt on the columns of m, projections applied twice."""
    cols = []
    for j in range(m.shape[-1]):
        v = m[..., j]
        for _ in range(2):
            for c in cols:
                v = v - c * (c * v).sum(-1, keepdim=True)
        cols.append(v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    return torch.stack(cols, -1)


def _quantize(S: torch.Tensor, q0: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Quantize a dense fp32 state [..., V, K] into a LeapQuant checkpoint."""
    state_scale = S.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    x = (S / state_scale).transpose(-1, -2)  # [..., K, V]
    qk = _mgs(x @ q0)  # [..., K, R]
    qv = _mgs(x.transpose(-1, -2) @ qk)  # [..., V, R]
    uk = x @ qv  # [..., K, R]
    compensator_scale = state_scale.sqrt()
    u_new = (uk.transpose(-1, -2) * compensator_scale).to(torch.float16)  # [..., R, K]
    q_new = (qv.transpose(-1, -2) * compensator_scale).to(torch.float16)  # [..., R, V]
    compensator = torch.einsum(
        "...rv,...rk->...vk",
        q_new.float() / compensator_scale,
        u_new.float() / compensator_scale,
    )
    e = x.transpose(-1, -2) - compensator
    s_k = e.abs().mean(dim=-2).sqrt().clamp_min(EPS)
    s_v = (e.abs() / s_k.unsqueeze(-2)).amax(dim=-1).clamp_min(EPS)
    codes = torch.round(e / (s_k.unsqueeze(-2) * s_v.unsqueeze(-1)) * QMAX)
    codes = codes.clamp(-QMAX, QMAX).to(torch.int8)
    return codes, s_k, s_v * state_scale.squeeze(-2), u_new, q_new


class GDNLeapQuantPools:
    """In-place composite-state views over the layer's ssm_state buffer.

    ``ssm_state`` is [num_slots, H, V, K] fp32 (possibly a strided view into a
    unified KV buffer). The first ``_LAYOUT_END`` bytes of each slot's storage
    hold the LeapQuant layout; the rest is unused.
    """

    def __init__(self, ssm_state: torch.Tensor):
        assert ssm_state.dtype == torch.float32, f"expect fp32 state, got {ssm_state.dtype}"
        if ssm_state.dim() == 4:
            self.num_slots, self.num_heads, self.head_v, self.head_k = ssm_state.shape
            slot_bytes = self.num_heads * self.head_v * self.head_k * 4
        else:
            self.num_slots, self.num_heads, _ = ssm_state.shape
            self.head_v = self.head_k = 128
            slot_bytes = self.num_heads * HEAD_BYTES
        H, V, K = self.num_heads, self.head_v, self.head_k
        self.slot_stride_bytes = ssm_state.stride(0) * 4
        assert self.slot_stride_bytes >= slot_bytes, "unexpected ssm_state slot stride"
        assert slot_bytes % HEAD_BYTES == 0, "slot storage must split into head blocks"
        assert H * HEAD_BYTES <= slot_bytes, "slot buffer too small for the head-block layout"
        i8 = ssm_state.view(torch.int8)
        self._state = ssm_state
        fv = i8.as_strided((self.num_slots, H, HEAD_BYTES), (self.slot_stride_bytes, HEAD_BYTES, 1))

        def typed(offset: int, dtype: torch.dtype, *tail: int) -> torch.Tensor:
            esize = torch.tensor([], dtype=dtype).element_size()
            n = 1
            tail_strides = []
            stride = 1
            for d in reversed(tail):
                tail_strides.insert(0, stride)
                n *= d
                stride *= d
            cols = fv[:, :, offset : offset + n * esize].view(dtype)
            strides = (fv.stride(0) // esize, fv.stride(1) // esize, *tail_strides)
            return cols.as_strided((self.num_slots, H, *tail), strides)

        self.codes = typed(_OFF_CODES, torch.int8, V, K)
        self.kbuf = typed(_OFF_KBUF, torch.bfloat16, WINDOW, K)
        self.ubuf = typed(_OFF_UBUF, torch.bfloat16, WINDOW, V)
        self.u = typed(_OFF_U, torch.float16, NUM_COMP_TOKENS, K)
        self.q = typed(_OFF_Q, torch.float16, NUM_COMP_TOKENS, V)
        self.s_k = typed(_OFF_SK, torch.float32, K)
        self.s_v = typed(_OFF_SV, torch.float32, V)
        self.w = typed(_OFF_W, torch.float32, WINDOW)
        p = fv[:, :, _OFF_P : _OFF_P + 4].view(torch.float32)
        self.p = p.as_strided((self.num_slots, H), (fv.stride(0) // 4, fv.stride(1) // 4))
        fill = fv[:, :, _OFF_FILL : _OFF_FILL + 4].view(torch.int32)
        self.fill = fill.as_strided((self.num_slots, H), (fv.stride(0) // 4, fv.stride(1) // 4))
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
        # fill == L (window full, flush already folded it): new-window start
        f = torch.where(f >= L, torch.zeros_like(f), f)
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
        # paper semantics: fill stays L after the fold; the next decode
        # step treats fill == L as the new-window start. Flush must not
        # clear it (the old zeroing double-folded nothing but hid the
        # host-ordering contract).

    def reconstruct_dense(self, idx: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full state, including live window updates."""
        t_idx, h_idx = self.th_index(idx)
        codes = self.codes[t_idx, h_idx].float()
        s_k = self.s_k[t_idx, h_idx]
        s_v = self.s_v[t_idx, h_idx]
        s0 = torch.einsum("bhrv,bhrk->bhvk", self.q[t_idx, h_idx].float(), self.u[t_idx, h_idx].float())
        s0 = s0 + codes * (s_k / QMAX).unsqueeze(-2) * s_v.unsqueeze(-1)
        f = self.fill[t_idx, h_idx]
        f = torch.where(f >= WINDOW, torch.zeros_like(f), f)
        j = torch.arange(WINDOW, device=idx.device)
        live = (j.view(1, 1, WINDOW) < f.unsqueeze(-1)).unsqueeze(-1)
        zeros = torch.zeros((), device=idx.device)
        ubuf = torch.where(live, self.ubuf[t_idx, h_idx].float(), zeros)
        kbuf = torch.where(live, self.kbuf[t_idx, h_idx].float(), zeros)
        return self.p[t_idx, h_idx].unsqueeze(-1).unsqueeze(-1) * s0 + torch.einsum(
            "bhj,bhjv,bhjk->bhvk", self.w[t_idx, h_idx], ubuf, kbuf
        )

    @torch.no_grad()
    def init_from_dense(self, ssm_state: torch.Tensor, idx: torch.Tensor) -> None:
        """Quantize the dense fp32 states of slots idx into composite checkpoints.

        Bumps ``_lq_epoch``: slot recycling invalidates any host-side
        fill mirror (flush-v2 due gate), which must then conservatively
        re-learn the affected slots.

        Must be called right after a dense-path write to those slots; reads the
        dense values via a gather first, so in-place aliasing is safe.
        """
        self._lq_epoch = getattr(self, "_lq_epoch", 0) + 1
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

    @torch.no_grad()
    def quantize_from_tensor(self, dense: torch.Tensor, idx: torch.Tensor) -> None:
        """Quantize a transient dense tensor and reset the window metadata."""
        self._lq_epoch = getattr(self, "_lq_epoch", 0) + 1
        t_idx, h_idx = self.th_index(idx)
        c, sk, sv, u_new, q_new = _quantize(dense.float(), self.q0)
        self.codes[t_idx, h_idx] = c
        self.s_k[t_idx, h_idx] = sk
        self.s_v[t_idx, h_idx] = sv
        self.u[t_idx, h_idx] = u_new
        self.q[t_idx, h_idx] = q_new
        self.w[t_idx, h_idx] = 0.0
        self.p[t_idx, h_idx] = 1.0
        self.fill[t_idx, h_idx] = 0
        self.kbuf[t_idx, h_idx] = 0.0
        self.ubuf[t_idx, h_idx] = 0.0

    def decode_state(self, idx: torch.Tensor) -> torch.Tensor:
        """Dense fp32 reconstruction of the checkpoint part (window buffer excluded)."""
        t_idx, h_idx = self.th_index(idx)
        s = torch.einsum("bhrv,bhrk->bhvk", self.q[t_idx, h_idx].float(), self.u[t_idx, h_idx].float())
        return s + self.codes[t_idx, h_idx].float() * (self.s_k[t_idx, h_idx] / QMAX).unsqueeze(
            -2
        ) * self.s_v[t_idx, h_idx].unsqueeze(-1)
