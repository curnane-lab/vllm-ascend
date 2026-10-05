import argparse
import json
import os
import time

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/workspace/models/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--out", default=None, help="jsonl dump path")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=float(os.getenv("SMOKE_MEM_UTIL", "0.85")),
        limit_mm_per_prompt={"image": 0},
        enable_prefix_caching=not args.no_prefix_caching,
        max_num_seqs=args.max_num_seqs,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [
        "请用中文简要介绍量子计算的基本原理。",
        "Write a short Python function to compute the nth Fibonacci number and explain it.",
        "Solve step by step: A train travels 300 km in 2 hours, then 150 km in 1 hour. What is its average speed for the whole journey?",
        "Explain the difference between TCP and UDP to a junior engineer, with one use case each.",
    ]
    if args.max_tokens >= 512:
        prompts = [
            "请详细推导线性回归的闭式解（正规方程），并解释每一步的数学原理。",
            "Write a detailed technical explanation of how the RSA algorithm works, including key generation, encryption, and decryption, with a small worked example.",
        ]
        if os.getenv("SMOKE_SINGLE_PROMPT"):
            prompts = prompts[:1]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    for o in outs:
        print("=" * 40)
        print(o.outputs[0].text)
    print(f"[smoke] {len(prompts)} prompts, {args.max_tokens} max tokens, {dt:.1f}s")
    if args.out:
        with open(args.out, "w") as f:
            for o in outs:
                f.write(
                    json.dumps(
                        {
                            "prompt": o.prompt,
                            "text": o.outputs[0].text,
                            "token_ids": list(o.outputs[0].token_ids),
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
