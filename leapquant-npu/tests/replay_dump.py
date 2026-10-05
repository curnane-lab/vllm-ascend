"""Offline replay of dumped decode steps: validate recurrence convention and
compare LeapQuant outputs against dense outputs on real-weight inputs."""

import glob
import os
import sys

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402

D = "/tmp/lq_dump"
layers = sorted(
    {os.path.basename(p)[6:].split("_step")[0] for p in glob.glob(f"{D}/dense_*_step0.pt")}
)


def load(prefix, tag, step):
    return torch.load(f"{D}/{tag}_{prefix}_step{step}.pt", weights_only=False)


def replay(prefix):
    # dense0 in the dense dump is now captured BEFORE the kernel runs
    d0 = load(prefix, "dense", 0)
    idx, dense0 = d0["idx"], d0["dense0"].cpu().float()
    H, V, K = dense0.shape[1:]
    rep = H // d0["query"].shape[1]
    scale = K ** -0.5

    S = dense0.clone()
    n_slots = 1700
    buf = torch.zeros(n_slots, *dense0.shape[1:], dtype=torch.float32)
    buf[idx] = dense0
    pools = GDNLeapQuantPools(buf)
    pools.init_from_dense(buf, idx)

    dense_rel = []
    lq_vs_dense = []
    lq_self = []
    prev_idx = idx
    steps = len(glob.glob(f"{D}/dense_{prefix}_step*.pt"))
    for t in range(steps):
        d = load(prefix, "dense", t)
        l = load(prefix, "lq", t)
        idx_t = d["idx"]
        if idx_t.shape != prev_idx.shape:
            break
        # emulate the framework's byte-copy state migration between slots
        moved = idx_t != prev_idx
        if moved.any():
            print(f"  [{prefix[-20:]}] step {t}: state migrated {prev_idx.tolist()} -> {idx_t.tolist()}")
            buf[idx_t[moved]] = buf[prev_idx[moved]]
        idx = idx_t
        kn16 = d["key"].cpu().float()
        qn16 = d["query"].cpu().float()
        kn = kn16.repeat_interleave(rep, dim=1)
        qn = qn16.repeat_interleave(rep, dim=1)
        v = d["value"].cpu().float()
        g = d["g"].cpu().float()
        beta = d["beta"].cpu().float()
        at = torch.exp(g)

        # dense oracle
        S_dec = at.view(-1, H, 1, 1) * S
        u = beta.unsqueeze(-1) * (v - torch.einsum("bhvk,bhk->bhv", S_dec, kn))
        o_dense = torch.einsum("bhvk,bhk->bhv", S_dec, qn) * scale + (kn * qn * scale).sum(-1, keepdim=True) * u
        S = S_dec + torch.einsum("bhv,bhk->bhvk", u, kn)

        out_dense = d["out"].cpu().float()
        dense_rel.append(((o_dense - out_dense).abs().mean() / out_dense.abs().mean()).item())

        o_lq = pools.decode_step(
            idx,
            kn.to(torch.bfloat16),
            qn.to(torch.bfloat16),
            v.to(torch.bfloat16),
            g,
            beta.to(torch.bfloat16),
            scale,
        )
        pools.flush(idx)
        out_lq = l["out"].cpu().float()
        lq_self.append(((o_lq.float() - out_lq).abs().mean() / out_lq.abs().mean().clamp_min(1e-6)).item())
        lq_vs_dense.append(((o_lq.float() - out_dense).abs().mean() / out_dense.abs().mean().clamp_min(1e-6)).item())
    return dense_rel, lq_self, lq_vs_dense


for prefix in layers[:4]:
    dense_rel, lq_self, lq_vs_dense = replay(prefix)
    short = prefix.split("layers_")[1].replace("_linear_attn", "")
    print(
        f"layer {short}: dense-replay rel err max={max(dense_rel):.5f} | "
        f"lq-self max={max(lq_self):.5f} | lq-vs-dense max={max(lq_vs_dense):.5f} "
        f"last={lq_vs_dense[-1]:.5f}"
    )
