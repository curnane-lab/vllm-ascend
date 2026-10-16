"""Profile one 8198-token prompt and 32 greedy decode tokens per mode."""

import argparse
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


BASE_PREFIX = (
    "以下是某分布式系统的设计文档。"
    "系统设计目标：高吞吐、低延迟、可水平扩展，失效隔离，数据一致。"
    "请记住，本系统采用分层一致性协议，并在日志中保留租约版本。"
) * 64


def build_prompt():
    return (
        BASE_PREFIX
        + "\n专属上下文编号 0：节点 0 的租约版本为 1000，副本数为 1。"
        + "\n请简要总结第 0 号节点的关键点。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, required=True)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--mem-util", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--max-batched-tokens", type=int, default=16384)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.max_num_batched_tokens = args.max_batched_tokens

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("LQ_MODEL_PATH", "/data1/models/Qwen3.5-9B"))
    global BASE_PREFIX
    while len(tokenizer.encode(BASE_PREFIX)) < 8192:
        BASE_PREFIX += "日志租约版本必须按节点递增，并在副本间保持一致。"
    prompt = build_prompt()
    prompt_tokens = len(tokenizer.encode(prompt))
    os.environ["VLLM_ASCEND_GDN_LQ_PROFILE"] = "1"
    os.environ["VLLM_ASCEND_GDN_LEAPQUANT"] = str(args.mode)
    max_model_len = max(args.max_model_len, prompt_tokens + args.decode)
    llm = LLM(
        model=os.environ.get("LQ_MODEL_PATH", "/data1/models/Qwen3.5-9B"),
        max_model_len=max_model_len,
        gpu_memory_utilization=min(args.mem_util, 0.3),
        limit_mm_per_prompt={"image": 0},
        enable_prefix_caching=bool(os.getenv("VLLM_ASCEND_MAMBA_UNEQUAL_PAGES")),
        max_num_seqs=1,
        block_size=128 if os.getenv("VLLM_ASCEND_MAMBA_UNEQUAL_PAGES") else None,
        max_num_batched_tokens=args.max_batched_tokens,
        enforce_eager=True,
        enable_chunked_prefill=bool(os.getenv("VLLM_ASCEND_MAMBA_UNEQUAL_PAGES")),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.decode)
    started = time.perf_counter()
    try:
        outputs = llm.generate([prompt], sp, use_tqdm=False)
    except Exception as exc:
        row = {"mode": args.mode, "error": repr(exc)}
        with open(args.out, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        raise
    wall = time.perf_counter() - started

    profile = None
    try:
        from vllm_ascend.ops import gdn as gdn_module
        profile = gdn_module._LQ_PROFILER.summarize()
    except Exception as exc:
        profile = {"error": repr(exc)}
    row = {
        "mode": args.mode,
        "prefix_len_target": 8192,
        "prefix_len_tokens": prompt_tokens,
        "decode": args.decode,
        "wall_s": round(wall, 6),
        "tok_per_s": round(sum(len(o.outputs[0].token_ids) for o in outputs) / wall, 6),
        "token_ids": list(outputs[0].outputs[0].token_ids),
        "profile": profile,
    }
    with open(args.out, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
