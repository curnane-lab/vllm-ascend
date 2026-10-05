"""Long-rollout state-error curve (paper Fig. 3 analogue) on real weights.

Replays the dumped fp32-run decode inputs through the LeapQuant pools and
measures, at every captured window boundary, the relative reconstruction error
|decode(checkpoint) - S_dense| / |S_dense|. Per-step quantization (the failure
mode LeapQuant avoids) grows this error linearly; per-window quantization keeps
it flat across windows.

Prereq: a dense-run dump with
  VLLM_ASCEND_GDN_LQ_DUMP=1 VLLM_ASCEND_GDN_LQ_DUMP_STEPS=2048 \
  VLLM_ASCEND_GDN_LQ_DUMP_LAYERS=<layers> VLLM_ASCEND_GDN_LQ_DUMP_STATE=1 \
  python3 scripts/smoke_qwen35.py --max-tokens <steps+margin> --out /tmp/x.jsonl
"""

import glob
import json
import os
import sys

import torch

sys.path.insert(0, "/mnt/workspace/vllm-ascend")
from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW  # noqa: E402

D = "/tmp/lq_dump"
DEVICE = "npu"


def layer_prefixes():
    out = set()
    for p in glob.glob(f"{D}/dense_*_step0.pt"):
        name = os.path.basename(p)
        out.add(name[len("dense_") :][: -len("_step0.pt")])
    return sorted(out)


def replay_layer(prefix):
    d0 = torch.load(f"{D}/dense_{prefix}_step0.pt", weights_only=False)
    idx, dense0 = d0["idx"].to(DEVICE), d0["dense0"].to(DEVICE).float()
    H = dense0.shape[1]
    rep = H // d0["query"].shape[1]
    steps = len(glob.glob(f"{D}/dense_{prefix}_step*.pt"))
    buf = torch.zeros(1700, *dense0.shape[1:], dtype=torch.float32, device=DEVICE)
    buf[idx] = dense0
    pools = GDNLeapQuantPools(buf.to(DEVICE))
    pools.init_from_dense(buf, idx)

    curve = []
    for t in range(steps):
        d = torch.load(f"{D}/dense_{prefix}_step{t}.pt", weights_only=False)
        kn = d["key"].to(DEVICE).float().repeat_interleave(rep, dim=1)
        qn = d["query"].to(DEVICE).float().repeat_interleave(rep, dim=1)
        v = d["value"].to(DEVICE)
        g = d["g"].to(DEVICE).float()
        beta = d["beta"].to(DEVICE)
        pools.decode_step(idx, kn.to(torch.bfloat16), qn.to(torch.bfloat16), v, g, beta, 1.0)
        pools.flush(idx)
        sp = f"{D}/state_{prefix}_step{t}.pt"
        if os.path.exists(sp):
            ref = torch.load(sp, weights_only=False)["state"].to(DEVICE).float()
            rec = pools.decode_state(idx)
            rel = ((rec - ref).abs().mean(dim=(1, 2, 3)) / ref.abs().mean(dim=(1, 2, 3)).clamp_min(1e-6))
            curve.append(
                {
                    "step": t + 1,
                    "rel_err_mean": rel.mean().item(),
                    "rel_err_max": rel.max().item(),
                }
            )
    return curve


def main():
    out = {}
    for prefix in layer_prefixes():
        short = prefix.split("layers_")[1].replace("_linear_attn", "")
        curve = replay_layer(prefix)
        out[short] = curve
        errs = [c["rel_err_mean"] for c in curve]
        if errs:
            print(
                f"layer {short}: windows={len(errs)} first={errs[0]:.5f} "
                f"last={errs[-1]:.5f} max={max(errs):.5f}"
            )
            for c in curve[::4]:
                print(f"  step {c['step']:5d}: mean={c['rel_err_mean']:.5f} max={c['rel_err_max']:.5f}")
    with open("/mnt/workspace/leapquant-npu/runs/long_rollout_curve.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
