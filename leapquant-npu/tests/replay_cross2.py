"""Cross-backend divergence on real dump data, per-step idx (batch may shrink)."""
import glob, sys
import torch
sys.path.insert(0, "/mnt/workspace/vllm-ascend")
sys.path.insert(0, "/mnt/workspace/leapquant-npu")
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools
from vllm_ascend.ops.gdn_lq_flush_ascendc import flush_pools_ascendc

D = "/tmp/lq_dump"

def load(p, t):
    return torch.load(f"{D}/{p}_step{t}.pt", weights_only=False)

def run_pair(layer):
    prefix = f"language_model_model_layers_{layer}_linear_attn"
    d0 = load(f"dense_{prefix}", 0)
    idx0, dense0 = d0["idx"].npu(), d0["dense0"].npu().float()
    H = dense0.shape[1]
    rep = H // d0["query"].shape[1]
    scale = 128 ** -0.5
    pools = []
    for _ in range(2):
        buf = torch.zeros(1700, H, 128, 128, device="npu")
        buf[idx0] = dense0
        p = GDNLeapQuantPools(buf)
        p.init_from_dense(buf, idx0)
        pools.append(p)
    n = len(glob.glob(f"{D}/dense_{prefix}_step*.pt"))
    for t in range(n):
        d = load(f"dense_{prefix}", t)
        idx = d["idx"].npu()
        kn = d["key"].npu().float().repeat_interleave(rep, 1).bfloat16()
        qn = d["query"].npu().float().repeat_interleave(rep, 1).bfloat16()
        outs = [p.decode_step(idx, kn, qn, d["value"].npu(),
                              d["g"].npu().float(), d["beta"].npu(), scale)
                for p in pools]
        flush_pools_ascendc(pools[1], idx)
        pools[0].flush(idx)
        out_dense = d["out"].npu().float()
        o_rel = ((outs[1].float() - outs[0].float()).abs().mean()
                 / outs[0].float().abs().mean().clamp_min(1e-6)).item()
        e_t = ((outs[0].float() - out_dense).abs().mean()
               / out_dense.abs().mean().clamp_min(1e-6)).item()
        e_a = ((outs[1].float() - out_dense).abs().mean()
               / out_dense.abs().mean().clamp_min(1e-6)).item()
        if (t + 1) % 16 == 0 or t >= n - 2:
            a = pools[0].decode_state(idx)
            b = pools[1].decode_state(idx)
            s_rel = ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-6)).item()
            print(f"  step {t:2d} idx={idx.tolist()}: x-rel={o_rel:.6f} "
                  f"vs-dense torch={e_t:.5f} ascendc={e_a:.5f} state-rel={s_rel:.6f}")

for layer in sys.argv[1:] or ["0", "10"]:
    print(f"layer {layer}:")
    run_pair(int(layer))
