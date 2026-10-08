"""M5 capacity ramp: max concurrent long-prefix requests per mode.

Each request: shared `prefix_len`-token prefix (cached after the first)
+ a short unique suffix + `decode` tokens of generation. N requests in
one batch; the engine queues what the pool cannot admit - the capacity
boundary shows as a TPOT cliff / timeout failures.

Emits one jsonl row per (mode, N, round) with throughput, mean TPOT
and completion status. Usage:
  python scripts/bench_capacity.py --mode 3 --prefix-len 4096 \
      --concurrency 64 --rounds 3 --out runs/m5_cap.jsonl
"""

import argparse
import json
import os
import time

from vllm import LLM, SamplingParams

PREFIX = (
    "以下是某分布式系统的详细设计文档，请基于它回答问题。\n"
    + "系统设计目标：高吞吐、低延迟、可水平扩展。"
    * 40  # ~160 tokens per rep -> tune below for target length
)


def build_prompts(n, prefix_len, tokenizer, unique=False):
    body = "系统设计目标：高吞吐、低延迟、可水平扩展，失效隔离，数据一致。" * 6
    # iterate until the tokenized prefix reaches the target length
    doc = ""
    while len(tokenizer.encode(doc)) < prefix_len:
        doc += body
    prompts = []
    for i in range(n):
        tag = f"编号 {i} 专属段落：节点 {i} 的本地日志与邻居表。" * (i + 1 if unique else 1)
        prompts.append(doc + tag + f"\n第 {i} 号报告：请用一段话总结上述文档的关键点。")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True)
    ap.add_argument("--prefix-len", type=int, default=4096)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--unique-prefix", action="store_true")
    ap.add_argument("--out", default="runs/m5_cap.jsonl")
    args = ap.parse_args()

    os.environ["VLLM_ASCEND_GDN_LEAPQUANT"] = str(args.mode)

    llm = LLM(
        model=os.environ.get("LQ_MODEL_PATH", "/mnt/workspace/models/Qwen3.5-9B"),
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.mem_util,
        limit_mm_per_prompt={"image": 0},
        enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()
    prompts = build_prompts(args.concurrency, args.prefix_len, tok,
                            unique=args.unique_prefix)
    if args.unique_prefix:
        # unique prefixes defeat the prefix cache: every request pays its
        # own blocks -> the pool (not the scheduler) is the binding limit.
        # The cache is configured at LLM() time; emulate by making every
        # prefix unique (done in build_prompts) and reporting the mode.
        pass
    sp = SamplingParams(temperature=0.0, max_tokens=args.decode)

    for r in range(args.rounds):
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        # TPOT: per-request decode latency after first token
        tpots = []
        ok = True
        for o in outs:
            ft = o.metrics.first_token_latency if hasattr(o.metrics, "first_token_latency") else None
            e2e = o.metrics.e2e_latency if hasattr(o.metrics, "e2e_latency") else None
            if e2e is None or len(o.outputs[0].token_ids) < args.decode:
                ok = False
            if ft is not None and e2e is not None and len(o.outputs[0].token_ids) > 1:
                tpots.append((e2e - ft) / (len(o.outputs[0].token_ids) - 1) * 1e3)
        row = {
            "mode": args.mode, "prefix_len": args.prefix_len,
            "concurrency": args.concurrency, "round": r + 1,
            "wall_s": round(dt, 2), "output_toks": ntok,
            "tok_per_s": round(ntok / dt, 1),
            "all_complete": ok,
            "mean_tpot_ms": round(sum(tpots) / len(tpots), 2) if tpots else None,
        }
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
