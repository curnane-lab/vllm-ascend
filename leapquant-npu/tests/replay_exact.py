"""Exact fp32 delta-rule recurrence from dense0 vs dump reference outputs."""
import glob, sys
import torch
D = "/tmp/lq_dump"

def load(p, t):
    return torch.load(f"{D}/{p}_step{t}.pt", weights_only=False)

for layer in sys.argv[1:] or ["0", "10"]:
    prefix = f"language_model_model_layers_{layer}_linear_attn"
    d0 = load(f"dense_{prefix}", 0)
    idx = d0["idx"]
    S = d0["dense0"].clone().double()          # [B,H,V,K] exact state
    rep = S.shape[1] // d0["query"].shape[1]
    scale = 128 ** -0.5
    errs = []
    n = len(glob.glob(f"{D}/dense_{prefix}_step*.pt"))
    for t in range(n):
        d = load(f"dense_{prefix}", t)
        kn = d["key"].double().repeat_interleave(rep, 1)      # [B,H,K]
        qn = d["query"].double().repeat_interleave(rep, 1)
        v = d["value"].double()
        g = d["g"].double().unsqueeze(-1).unsqueeze(-1)       # [B,H,1,1]
        beta = d["beta"].double().unsqueeze(-1)               # [B,H,1]
        upd = beta * (v - torch.einsum("bhvk,bhk->bhv", S, kn))
        S = torch.exp(g) * S + torch.einsum("bhv,bhk->bhvk", upd, kn)
        o = torch.einsum("bhvk,bhk->bhv", S, qn) * scale
        out_dense = d["out"].double()
        errs.append(((o - out_dense).abs().mean()
                     / out_dense.abs().mean().clamp_min(1e-6)).item())
    print(f"layer {layer}: exact-recurrence vs dump out: "
          f"max={max(errs):.6f} err@[15,16,31,36,37,38]={['%.5f' % errs[i] for i in (15, 16, 31, 36, 37, 38)]}")
