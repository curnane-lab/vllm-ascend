#!/usr/bin/env python3
"""End-to-end vLLM benchmark with request metrics and optional dump."""

import argparse
import json
import os
import re
import subprocess
import time

from vllm import LLM, SamplingParams

SHORT_PROMPTS = [
    "请用中文简要介绍量子计算的基本原理。",
    "Write a short Python function to compute the nth Fibonacci number and explain it.",
    "Solve step by step: A train travels 300 km in 2 hours, then 150 km in 1 hour. What is its average speed for the whole journey?",
    "Explain the difference between TCP and UDP to a junior engineer, with one use case each.",
]
LONG_PROMPTS = [
    "请详细推导线性回归的闭式解（正规方程），并解释每一步的数学原理。",
    "Write a detailed technical explanation of how the RSA algorithm works, including key generation, encryption, and decryption, with a small worked example.",
]


def metric_value(metric, name):
    value = getattr(metric, name, None)
    if value is None and isinstance(metric, dict):
        value = metric.get(name)
    return value


def parse_pool_gib():
    try:
        text = subprocess.run(["npu-smi", "info"], capture_output=True,
                              text=True, timeout=30).stdout
        used = [int(m.group(1)) for m in
                re.finditer(r"(\d+)\s*/\s*65536", text)]
        return max(used) / 1024 if used else None
    except Exception:
        return None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LQ_MODEL_PATH", "/data1/models/Qwen3.5-9B"))
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--shared-prefix", type=int, default=0)
    parser.add_argument("--requests", type=int, default=0, help="repeat prompts to this count")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--dump", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    base_prompts = LONG_PROMPTS if args.long else SHORT_PROMPTS
    if args.shared_prefix > 0:
        filler = "分布式系统关注吞吐、延迟、一致性和失效隔离。" * 12
        while len(filler) < args.shared_prefix * 2:
            filler += filler
        base_prompts = [filler[: args.shared_prefix * 2] + p for p in base_prompts]
    prompts = list(base_prompts)
    while len(prompts) < args.requests:
        prompts.extend(base_prompts)
    prompts = prompts[: args.requests or len(base_prompts)]

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt={"image": 0, "video": 0},
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.output_tokens)
    if args.warmup:
        llm.generate(prompts[: min(len(prompts), 2)],
                     SamplingParams(temperature=0.0, max_tokens=8),
                     use_tqdm=False)

    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    wall = time.perf_counter() - started

    output_tokens = 0
    ttfts, e2es, tpots, itls = [], [], [], []
    records = []
    for index, output in enumerate(outputs):
        tokens = list(output.outputs[0].token_ids)
        output_tokens += len(tokens)
        metrics = output.metrics
        first = metric_value(metrics, "first_token_latency")
        e2e = metric_value(metrics, "e2e_latency")
        if first is not None:
            ttfts.append(float(first))
        if e2e is not None:
            e2es.append(float(e2e))
        if first is not None and e2e is not None and len(tokens) > 1:
            tpots.append((float(e2e) - float(first)) / (len(tokens) - 1) * 1000.0)
        records.append({
            "index": index,
            "prompt_token_ids": list(output.prompt_token_ids),
            "token_ids": tokens,
            "text": output.outputs[0].text,
            "first_token_latency": first,
            "e2e_latency": e2e,
            "finish_reason": str(output.outputs[0].finish_reason),
        })

    def stats(values, scale=1.0):
        return {
            "mean": sum(values) / len(values) * scale if values else None,
            "p50": percentile(values, 0.50) * scale if values else None,
            "p90": percentile(values, 0.90) * scale if values else None,
            "p99": percentile(values, 0.99) * scale if values else None,
        }

    row = {
        "tag": args.tag,
        "mode": int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0")),
        "step_v2": int(os.getenv("VLLM_ASCEND_GDN_LQ_STEP_V2", "0")),
        "requests": len(prompts),
        "output_tokens": args.output_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "prefix_caching": args.enable_prefix_caching,
        "long": args.long,
        "shared_prefix_tokens_estimate": args.shared_prefix,
        "wall_s": wall,
        "generated_tokens": output_tokens,
        "completion_ratio": output_tokens / (args.output_tokens * len(prompts)),
        "request_throughput": len(prompts) / wall,
        "output_token_throughput": output_tokens / wall,
        "ttft_s": stats(ttfts),
        "e2e_s": stats(e2es),
        "tpot_ms": stats(tpots),
        "pool_gib_at_end": parse_pool_gib(),
    }
    with open(args.out, "a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))
    if args.dump:
        with open(args.dump, "w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
