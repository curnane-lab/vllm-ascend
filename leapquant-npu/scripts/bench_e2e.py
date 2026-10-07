"""End-to-end decode throughput benchmark: fp32 baseline vs LeapQuant modes.

Sends B copies of one prompt (greedy) and measures generation wall time.
Mode is selected by VLLM_ASCEND_GDN_LEAPQUANT (0/2/3); step v2 by
VLLM_ASCEND_GDN_LQ_STEP_V2. Optional ACL graph via --cudagraph
piecewise|full (implies not eager).

  python3 scripts/bench_e2e.py --batch 16 --tokens 128
  VLLM_ASCEND_GDN_LEAPQUANT=3 python3 scripts/bench_e2e.py --batch 64 --tokens 128
  VLLM_ASCEND_GDN_LEAPQUANT=3 python3 scripts/bench_e2e.py --batch 16 \
      --cudagraph piecewise --tag m3-piecewise
"""

import argparse
import json
import os
import subprocess
import time

import torch
from vllm import LLM, SamplingParams

PROMPT = "请详细解释 Transformer 架构中注意力机制的计算过程，并逐步推导。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--eager", action="store_true", default=True)
    ap.add_argument("--graph", dest="eager", action="store_false")
    ap.add_argument("--cudagraph", choices=["none", "piecewise", "full"],
                    default="none")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0"))
    v2 = int(os.getenv("VLLM_ASCEND_GDN_LQ_STEP_V2", "0"))
    kwargs = {}
    if args.cudagraph != "none":
        kwargs["compilation_config"] = {
            "cudagraph_mode": args.cudagraph.upper(),
        }
        args.eager = False
    llm = LLM(
        model="/mnt/workspace/models/Qwen3.5-9B",
        max_model_len=4096,
        enforce_eager=args.eager,
        gpu_memory_utilization=float(os.getenv("SMOKE_MEM_UTIL", "0.85")),
        limit_mm_per_prompt={"image": 0},
        max_num_seqs=args.batch,
        enable_prefix_caching=False,
        **kwargs,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens)
    prompts = [PROMPT] * args.batch
    llm.generate(prompts, sp)  # compile/warm at the FULL batch shape
    torch.npu.reset_peak_memory_stats()
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    # wall includes the prefills of all prompts amortized over the whole
    # run; per-step figures derived as wall/tokens are "generation wall
    # amortized (prefill included)", NOT pure decode step latency
    # engine allocations live in the EngineCore process; torch stats in
    # the driver see nothing. Query the device directly while the LLM
    # (and its KV pool) is still alive.
    try:
        out = subprocess.run(
            ["npu-smi", "info"], capture_output=True,
            text=True, timeout=30).stdout
        import re as _re
        used = [int(m.group(1)) for m in
                _re.finditer(r"(\d+)\s*/\s*65536", out)]
        # single end-of-run sample: it bounds the pool size but can miss
        # the true transient peak; treat as "pool at end of run"
        peak = max(used) / 1024 if used else 0.0
    except Exception:
        peak = torch.npu.max_memory_allocated() / 2 ** 30
    # v2 branches: a silent triton/v1 fallback invalidates the number.
    # the engine's stderr shares our stream; scan the whole captured
    # output for fallback signatures (Task 1, grok plan).
    v2_env = bool(os.getenv("VLLM_ASCEND_GDN_LQ_STEP_V2")
                  or os.getenv("VLLM_ASCEND_GDN_LQ_FLUSH_V2"))
    fallback = False
    if v2_env:
        import io as _io
        # callers capture stdout; slow-path exceptions and the one-line
        # unavailability notice go to the shared stderr -> both appear
        # in the terminal/jsonl pipeline. Re-check via the vllm log is
        # overkill: require the caller to tee; here we assert on what
        # this process can see (driver-side import check).
        try:
            lq_path = os.environ.get("VLLM_ASCEND_GDN_LQ_PATH",
                                     "/mnt/workspace/leapquant-npu")
            sys.path.insert(0, lq_path)
            if os.getenv("VLLM_ASCEND_GDN_LQ_STEP_V2"):
                import gdn_lq_step_v2  # noqa: F401
        except ImportError:
            fallback = True
    line = (
        f"[bench-e2e] mode={mode} v2={v2} batch={args.batch} "
        f"tokens={args.tokens} eager={args.eager} cg={args.cudagraph} "
        f"wall={dt:.1f}s output_toks={ntok} throughput={ntok / dt:.1f} tok/s "
        f"peak_mem={peak:.2f}GiB"
    )
    if v2_env and fallback:
        line = "[bench-e2e] FAIL(v2 unavailable: import failed) " + line
    if args.tag:
        line += f" tag={args.tag}"
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps({
                "mode": mode, "v2": v2, "batch": args.batch,
                "tokens": args.tokens, "eager": args.eager,
                "cudagraph": args.cudagraph, "wall_s": dt,
                "output_toks": ntok, "tok_per_s": ntok / dt,
                "pool_at_end_gib": peak, "peak_note": "single npu-smi sample at end of run; not the transient peak", "tag": args.tag,
                "token_ids": [list(o.outputs[0].token_ids) for o in outs],
            }) + "\n")


if __name__ == "__main__":
    main()
