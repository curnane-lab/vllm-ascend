"""Cross-backend divergence on real dump data: ascendc vs torch flush."""
import glob, os, sys
from pathlib import Path
import torch
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
sys.path.insert(0, str(_REPO / "leapquant-npu"))
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools
from vllm_ascend.ops.gdn_lq_flush_ascendc import flush_pools_ascendc

D = "/tmp/lq_dump"

def load(p, t):
    return torch.load(f"{D}/{p}_step{t}.pt", weights_only=False)

def run_pair(layer):
    prefix = f"language_model_model_layers_{layer}_linear_attn"
    d0 = load(f"dense_{prefix}", 0)
    idx, dense0 = d0["idx"].npu(), d0["dense0"].npu().float()
    H = dense0.shape[1]
    rep = H // d0["query"].shape[1]
    scale = 128 ** -0.5
    bufs = [torch.zeros(1700, H, 128, 128, device="npu") for _ in range(2)]
    pools = []
    for buf in bufs:
        buf[idx] = dense0
        p = GDNLeapQuantPools(buf)
        p.init_from_dense(buf, idx)
        pools.append(p)
    n = len(glob.glob(f"{D}/dense_{prefix}_step*.pt"))
    o_diffs, s_diffs = [], []
    for t in range(n):
        d = load(f"dense_{prefix}", t)
        kn = d["key"].npu().float().repeat_interleave(rep, 1).bfloat16()
        qn = d["query"].npu().float().repeat_interleave(rep, 1).bfloat16()
        outs = []
        for p in pools:
            outs.append(pools_decode(p, idx, kn, qn, d, scale))
        flush_pools_ascendc(pools[1], idx)
        pools[0].flush(idx)
        out_dense = d["out"].npu().float()
        o_diffs.append(((outs[1].float() - outs[0].float()).abs().mean()
                        / outs[0].float().abs().mean().clamp_min(1e-6)).item())
        err_vs_dense = [((o.float() - out_dense).abs().mean()
                         / out_dense.abs().mean().clamp_min(1e-6)).item() for o in outs]
        if (t + 1) % 16 == 0 or t == n - 1:
            a = pools[0].decode_state(idx)
            b = pools[1].decode_state(idx)
            s_diffs.append((t, ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-6)).item()))
            print(f"  step {t:2d}: ascendc-vs-torch out rel={o_diffs[-1]:.6f} | "
                  f"vs dense: torch={err_vs_dense[0]:.5f} ascendc={err_vs_dense[1]:.5f} | "
                  f"state rel={s_diffs[-1][1]:.6f}")
    return o_diffs

def pools_decode(p, idx, kn, qn, d, scale):
    return p.decode_step(idx, kn, qn, d["value"].npu(),
                         d["g"].npu().float(), d["beta"].npu(), scale)

for layer in sys.argv[1:] or ["0", "10"]:
    print(f"layer {layer}:")
    run_pair(int(layer))
