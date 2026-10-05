"""Long-context (8K-64K) logprob drift: LeapQuant vs fp32 baseline.

Feeds a long fixed prompt in one prefill (requires max_num_batched_tokens >=
prompt length; the LeapQuant chunked-prefill guard is thus not triggered) and
compares per-position prompt logprobs between the fp32 baseline and the
LeapQuant run. This is the NPU analogue of LeapQuant paper Fig. 3 over context
length: quantization error should stay flat (per-window scheme) rather than
grow linearly (per-step scheme).

Usage:
  python3 scripts/measure_drift_long.py --mode fp32   --out runs/driftl_fp32.json
  VLLM_ASCEND_GDN_LEAPQUANT=1 python3 scripts/measure_drift_long.py --mode leapquant --out runs/driftl_lq.json
  python3 scripts/measure_drift_long.py --mode bf16 --out runs/driftl_bf16.json
  python3 scripts/compare_drift_long.py runs/driftl_fp32.json runs/driftl_lq.json
"""

import argparse
import json
import math

from vllm import LLM, SamplingParams

# ~10K tokens of mixed technical Chinese/English; repeated with section markers
# to reach the target length with non-trivial per-position distributions.
_SECTION = (
    "## 循环状态量化笔记\n"
    "线性注意力把 token 历史压缩为固定大小的循环状态矩阵。每个解码步都要从显存读出状态、"
    "施加轻量更新再写回，算术强度极低，吞吐被显存带宽限制。量化状态可以自然降低带宽与容量，"
    "但面临两个障碍：舍入误差沿递推链相干累积，以及状态矩阵中少数离群行列主导量化范围。"
    "按窗口量化把最近十六个 rank-one 更新以高精度缓存，只在窗口末尾重建并量化一次，"
    "把量化频率降低一个数量级。补偿 token 以 rank-one 拟合保留主离群结构，"
    "残差平滑通过可逆对角缩放均衡行幅值，两者叠加把离群峰值压低两个数量级。\n"
    "## Recurrent state quantization notes\n"
    "Linear attention compresses the token history into a fixed-size recurrent state matrix. "
    "Each decoding step reads the state from memory, applies a lightweight update and writes it "
    "back, so throughput is bound by memory bandwidth. Quantizing the state reduces bandwidth and "
    "capacity, but faces two obstacles: rounding errors accumulate coherently along the recurrence, "
    "and a few outlier rows and columns of the state dominate the quantization range. Per-window "
    "quantization buffers the last sixteen rank-one updates in high precision and re-quantizes the "
    "state only once per window, reducing the quantization frequency by an order of magnitude. "
    "Compensator tokens retain the dominant outlier structure as a rank-one fit, and residual "
    "smoothing equalizes row magnitudes through an invertible diagonal scaling; together they "
    "suppress the outlier peak by two orders of magnitude.\n"
)


def build_text(target_tokens: int) -> str:
    # rough 1.6 tokens per CJK char, 0.75 per English word; the section is
    # ~700 CJK + ~120 English words ~ 1500 tokens; repeat with index markers
    parts = []
    i = 0
    total = 0
    while total < target_tokens:
        parts.append(f"[segment {i}]\n" + _SECTION)
        total += 1550
        i += 1
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "leapquant", "bf16"], default="fp32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=33000)
    ap.add_argument("--max-model-len", type=int, default=40960)
    args = ap.parse_args()

    kwargs = {}
    if args.mode == "bf16":
        kwargs["mamba_ssm_cache_dtype"] = "bfloat16"
    llm = LLM(
        model="/mnt/workspace/models/Qwen3.5-9B",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0},
        max_num_batched_tokens=args.max_model_len,
        enable_prefix_caching=False,
        **kwargs,
    )
    if args.mode == "leapquant":
        import os

        assert os.getenv("VLLM_ASCEND_GDN_LEAPQUANT") == "1", "set the env var"

    text = build_text(args.target_tokens)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    out = llm.generate([text], sp)
    lp = out[0].prompt_logprobs
    vals = []
    for pos in lp:
        if not pos:
            continue
        entry = next(iter(pos.values()))
        vals.append(float(entry.logprob) if hasattr(entry, "logprob") else float(entry[0]))
    with open(args.out, "w") as f:
        json.dump([v if math.isfinite(v) else None for v in vals], f)
    import torch

    v = torch.tensor([x if x is not None else float("nan") for x in vals])
    print(f"[drift-long:{args.mode}] tokens={len(vals)} mean_logprob={v.nanmean().item():.4f}")


if __name__ == "__main__":
    main()
