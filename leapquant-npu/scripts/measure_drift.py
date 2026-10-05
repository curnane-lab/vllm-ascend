"""Measure prompt-logprob drift of LeapQuant/BF16-state vs the fp32 baseline.

Feeds a long fixed prompt and compares per-token logprobs; the drift curve is
the NPU analogue of LeapQuant paper Fig. 3 (state error vs context length).
"""

import argparse
import json

import torch
from vllm import LLM, SamplingParams

DRIFT_TEXT = (
    "线性注意力通过将历史压缩为固定大小的循环状态来降低长上下文推理成本。"
    "本文研究该循环状态的量化方法，目标是降低逐 token 解码时的显存带宽占用。"
    "量化面临两个障碍：其一是舍入误差的递推累积，其二是状态矩阵中离群点行与列的存在。"
    "LeapQuant 提出按窗口量化、补偿 token 与残差平滑三个组件来缓解上述问题。"
    "按窗口量化将状态更新缓存为高精度 rank-one 更新，每十六个 token 才重建并量化一次，"
    "从而将量化频率降低一个数量级，并显著抑制长上下文下的误差增长。"
    "补偿 token 以 rank-one 拟合保留状态中的主导大幅值结构，残差平滑通过对角缩放平衡行幅值，"
    "二者共同将离群峰值从中位数的百余倍压低至个位数倍数。"
    "实验表明八比特状态在十二个模型任务组合上与 fp32 基线精度相当。"
    "Linear attention compresses token history into a fixed-size recurrent state to reduce "
    "the cost of long-context inference. This state must be read from and written to memory "
    "at every decoding step, which makes decoding bandwidth-bound. Quantizing the state to "
    "low precision is attractive but suffers from error accumulation along the recurrence and "
    "from outlier rows and columns that dominate the quantization scale. Per-window quantization "
    "buffers the rank-one updates of the last sixteen tokens in high precision and re-quantizes "
    "the state only once per window, reducing the quantization frequency and suppressing error "
    "growth. Compensator tokens keep the dominant rank-one structure in fp16 while residual "
    "smoothing equalizes row magnitudes through a diagonal scaling, bringing outliers from more "
    "than a hundred times the median down to a few times. Experiments across the Qwen, Kimi and "
    "GLM model families show that eight-bit states match fp32 accuracy on long reasoning and "
    "code generation tasks while reducing state memory by more than three fold and accelerating "
    "decoding by up to three point seven times at the kernel level. "
) * 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "leapquant", "bf16"], default="fp32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kwargs = {}
    if args.mode == "bf16":
        kwargs["mamba_ssm_cache_dtype"] = "bfloat16"
    llm = LLM(
        model="/mnt/workspace/models/Qwen3.5-9B",
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0},
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        **kwargs,
    )
    if args.mode == "leapquant":
        import os

        assert os.getenv("VLLM_ASCEND_GDN_LEAPQUANT") == "1"

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    out = llm.generate([DRIFT_TEXT], sp)
    lp = out[0].prompt_logprobs  # list over prompt positions of {tok: (logprob, ...)}
    vals = []
    for pos in lp:
        if not pos:
            continue
        entry = next(iter(pos.values()))
        if hasattr(entry, "logprob"):
            vals.append(float(entry.logprob))
        else:
            vals.append(float(entry[0] if isinstance(entry, (tuple, list)) else entry))
    import math

    with open(args.out, "w") as f:
        json.dump([v if math.isfinite(v) else None for v in vals], f)
    v = torch.tensor(vals)
    print(f"[drift:{args.mode}] tokens={len(vals)} mean_logprob={v.mean().item():.4f}")


if __name__ == "__main__":
    main()
