import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch_npu  # noqa: F401
from torch_npu.npu import mstx
from vllm import LLM, SamplingParams

from vllm_ascend.worker.worker import NPUWorker

PROMPT = "请详细解释 Transformer 架构中注意力机制的计算过程，并逐步推导。"


def _stats(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _generate(llm, sampling, batch):
    started = time.perf_counter()
    prompts = [f"{PROMPT} 请求编号 {index}。" for index in range(batch)]
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    wall = time.perf_counter() - started
    tpots = []
    ttfts = []
    for output in outputs:
        metrics = output.metrics
        tokens = len(output.outputs[0].token_ids)
        ttfts.append(float(metrics.first_token_latency) * 1e3)
        decode = float(metrics.last_token_ts) - float(metrics.first_token_ts)
        tpots.append(decode / (tokens - 1) * 1e3)
    return outputs, {
        "wall_s": wall,
        "generated_tokens": sum(len(output.outputs[0].token_ids) for output in outputs),
        "ttft_ms": _stats(ttfts),
        "tpot_ms": _stats(tpots),
    }


def _install_step_markers():
    original = NPUWorker.execute_model
    counter = {"value": 0}

    def execute_model(self, *args, **kwargs):
        step = counter["value"]
        counter["value"] += 1
        mstx.mark(f"GDN_BENCH_STEP_{step}_BEGIN")
        try:
            return original(self, *args, **kwargs)
        finally:
            mstx.mark(f"GDN_BENCH_STEP_{step}_END")

    NPUWorker.execute_model = execute_model


def _tensor_layout(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
        "data_ptr": tensor.data_ptr(),
        "is_contiguous": tensor.is_contiguous(),
    }


def _install_cache_view_debug():
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState

    original = AscendAttentionBackendImpl._get_fia_params
    seen = set()

    def get_fia_params(self, key, value, attn_metadata, kv_cache=None):
        result = original(self, key, value, attn_metadata, kv_cache)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly and self._layer_name not in seen:
            seen.add(self._layer_name)
            out_key, out_value, block_size = result[:3]
            print(
                "CACHE_VIEW_DEBUG "
                + json.dumps(
                    {
                        "layer": self._layer_name,
                        "layout": self.kv_cache_layout,
                        "key": _tensor_layout(key),
                        "value": _tensor_layout(value),
                        "fia_key": _tensor_layout(out_key),
                        "fia_value": _tensor_layout(out_value),
                        "block_size": block_size,
                    }
                ),
                flush=True,
            )
        return result

    AscendAttentionBackendImpl._get_fia_params = get_fia_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LQ_MODEL_PATH", "/data/f00538480/Qwen3.5-4B"))
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.trace_dir = args.trace_dir.resolve()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    _install_step_markers()
    if os.getenv("CACHE_VIEW_DEBUG"):
        _install_cache_view_debug()

    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_batched_tokens=4096,
        max_num_seqs=args.batch,
        enforce_eager=args.eager,
        compilation_config={"cudagraph_mode": "NONE" if args.eager else "FULL"},
        gpu_memory_utilization=float(os.getenv("SMOKE_MEM_UTIL", "0.75")),
        limit_mm_per_prompt={"image": 0},
        enable_prefix_caching=False,
        disable_log_stats=False,
    )

    def sampling(tokens):
        return SamplingParams(
            temperature=0.0,
            max_tokens=tokens,
            min_tokens=tokens,
            ignore_eos=True,
        )

    _generate(llm, sampling(args.warmup_tokens), args.batch)
    baseline_outputs, baseline = _generate(llm, sampling(args.tokens), args.batch)
    profile_outputs, profile = _generate(llm, sampling(args.tokens), args.batch)

    baseline_tokens = [list(output.outputs[0].token_ids) for output in baseline_outputs]
    profile_tokens = [list(output.outputs[0].token_ids) for output in profile_outputs]
    payload = {
        "schema_version": 1,
        "model": args.model,
        "batch": args.batch,
        "tokens": args.tokens,
        "decode_steps": args.tokens - 1,
        "mode": "dense MR V2, FULL graph, prefix cache off, fixed length",
        "token_trajectories_match": baseline_tokens == profile_tokens,
        "baseline": baseline,
        "profile": profile,
        "baseline_token_ids": baseline_tokens,
        "profile_token_ids": profile_tokens,
        "trace_dir": str(args.trace_dir),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
