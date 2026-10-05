"""End-to-end decode throughput benchmark: fp32 baseline vs LeapQuant modes.

Sends B copies of one prompt (greedy) and measures generation wall time.
Mode is selected by VLLM_ASCEND_GDN_LEAPQUANT (0/2/3).

  python3 scripts/bench_e2e.py --batch 16 --tokens 128
  VLLM_ASCEND_GDN_LEAPQUANT=3 python3 scripts/bench_e2e.py --batch 64 --tokens 128
"""

import argparse
import os
import time

from vllm import LLM, SamplingParams

PROMPT = "请详细解释 Transformer 架构中注意力机制的计算过程，并逐步推导。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--eager", action="store_true", default=True)
    ap.add_argument("--graph", dest="eager", action="store_false")
    args = ap.parse_args()

    mode = int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0"))
    llm = LLM(
        model="/mnt/workspace/models/Qwen3.5-9B",
        max_model_len=4096,
        enforce_eager=args.eager,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0},
        max_num_seqs=args.batch,
        enable_prefix_caching=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens)
    prompts = [PROMPT] * args.batch
    llm.generate(prompts[:1], sp)  # compile/warm
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"[bench-e2e] mode={mode} batch={args.batch} tokens={args.tokens} "
        f"eager={args.eager} wall={dt:.1f}s output_toks={ntok} "
        f"throughput={ntok / dt:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
