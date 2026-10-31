import argparse
import json
import os
import statistics
import threading
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops import gdn as ascend_gdn
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen_gdn
from vllm.forward_context import get_forward_context

PROMPT = "请详细解释 Transformer 架构中注意力机制的计算过程，并逐步推导。"
_LOCAL = threading.local()
_CONTROL = None
_WORKER_OUTPUT = None
_EXPECTED_CALLS = None
_RECORDS = None


def _is_pure_decode(layer):
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return False
    metadata = metadata.get(layer.prefix)
    return bool(
        metadata is not None
        and metadata.num_prefills == 0
        and metadata.num_decodes > 0
        and metadata.spec_sequence_masks is None
    )


def _active():
    return getattr(_LOCAL, "active", False)


def _event():
    return torch.npu.Event(enable_timing=True)


def _dump_worker_records():
    torch.npu.synchronize()
    result = {}
    for name, pairs in _RECORDS.items():
        times = [start.elapsed_time(end) for start, end in pairs]
        steps = [sum(times[i : i + 24]) for i in range(0, len(times), 24)]
        result[name] = {"call_ms": times, "step_total_ms": steps}
    temporary = _WORKER_OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(_WORKER_OUTPUT)


def _install_profilers():
    global _RECORDS
    _RECORDS = {name: [] for name in ("module", "core", "conv", "packed_recurrent")}
    original_forward = ascend_gdn.AscendGatedDeltaNetAttention.forward
    original_core = ascend_gdn.AscendGatedDeltaNetAttention._forward_core
    original_conv = qwen_gdn._fla_conv1d_update
    original_packed = qwen_gdn.fused_recurrent_gated_delta_rule_packed_decode

    def forward(self, *args, **kwargs):
        if _CONTROL is None or not _CONTROL.exists() or not _is_pure_decode(self):
            return original_forward(self, *args, **kwargs)
        start, end = _event(), _event()
        start.record()
        _LOCAL.active = True
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            end.record()
            _LOCAL.active = False
            _RECORDS["module"].append((start, end))
            if len(_RECORDS["module"]) == _EXPECTED_CALLS:
                _dump_worker_records()

    def core(self, *args, **kwargs):
        if not _active():
            return original_core(self, *args, **kwargs)
        start, end = _event(), _event()
        start.record()
        try:
            return original_core(self, *args, **kwargs)
        finally:
            end.record()
            _RECORDS["core"].append((start, end))

    def conv(*args, **kwargs):
        if not _active():
            return original_conv(*args, **kwargs)
        start, end = _event(), _event()
        start.record()
        try:
            return original_conv(*args, **kwargs)
        finally:
            end.record()
            _RECORDS["conv"].append((start, end))

    def packed(*args, **kwargs):
        if not _active():
            return original_packed(*args, **kwargs)
        start, end = _event(), _event()
        start.record()
        try:
            return original_packed(*args, **kwargs)
        finally:
            end.record()
            _RECORDS["packed_recurrent"].append((start, end))

    ascend_gdn.AscendGatedDeltaNetAttention.forward = forward
    ascend_gdn.AscendGatedDeltaNetAttention._forward_core = core
    qwen_gdn._fla_conv1d_update = conv
    qwen_gdn.fused_recurrent_gated_delta_rule_packed_decode = packed


def _stats(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _request_metrics(outputs):
    tpots = []
    ttfts = []
    for output in outputs:
        metrics = output.metrics
        tokens = len(output.outputs[0].token_ids)
        ttfts.append(float(metrics.first_token_latency) * 1e3)
        decode = float(metrics.last_token_ts) - float(metrics.first_token_ts)
        tpots.append(decode / (tokens - 1) * 1e3)
    return {
        "ttft_ms": _stats(ttfts),
        "tpot_ms": _stats(tpots),
    }


def _generate(llm, sampling, batch):
    started = time.perf_counter()
    outputs = llm.generate([PROMPT] * batch, sampling, use_tqdm=False)
    wall = time.perf_counter() - started
    return outputs, {
        "wall_s": wall,
        "generated_tokens": sum(len(output.outputs[0].token_ids) for output in outputs),
        "metrics": _request_metrics(outputs),
    }


def main():
    global _CONTROL, _WORKER_OUTPUT, _EXPECTED_CALLS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LQ_MODEL_PATH", "/data/f00538480/Qwen3.5-4B"))
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()

    from vllm import LLM, SamplingParams

    _EXPECTED_CALLS = 24 * (args.tokens - 1)
    _CONTROL = args.output.with_suffix(".enable")
    _WORKER_OUTPUT = args.output.with_suffix(".worker.json")
    for path in (_CONTROL, _WORKER_OUTPUT):
        path.unlink(missing_ok=True)
    _install_profilers()

    llm = LLM(
        model=args.model,
        max_model_len=512,
        max_num_batched_tokens=4096,
        max_num_seqs=args.batch,
        enforce_eager=True,
        gpu_memory_utilization=float(os.getenv("SMOKE_MEM_UTIL", "0.80")),
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
    _CONTROL.touch()
    try:
        profile_outputs, profile = _generate(llm, sampling(args.tokens), args.batch)
    finally:
        _CONTROL.unlink(missing_ok=True)

    if not _WORKER_OUTPUT.exists():
        raise RuntimeError("EngineCore did not produce GDN event records")
    events = json.loads(_WORKER_OUTPUT.read_text())
    expected = 24 * (args.tokens - 1)
    for name, record in events.items():
        if len(record["call_ms"]) != expected:
            raise RuntimeError(f"{name}: expected {expected} calls, got {len(record['call_ms'])}")
        record["call_stats"] = _stats(record["call_ms"])
        record["step_stats"] = _stats(record["step_total_ms"])

    baseline_tokens = [list(output.outputs[0].token_ids) for output in baseline_outputs]
    profile_tokens = [list(output.outputs[0].token_ids) for output in profile_outputs]
    if baseline_tokens != profile_tokens:
        raise RuntimeError("baseline and profile token trajectories differ")

    tpot = baseline["metrics"]["tpot_ms"]["mean"]
    packed = events["packed_recurrent"]["step_stats"]["mean"]
    module = events["module"]["step_stats"]["mean"]
    core = events["core"]["step_stats"]["mean"]
    conv = events["conv"]["step_stats"]["mean"]
    remaining = tpot - packed
    payload = {
        "schema_version": 1,
        "model": args.model,
        "batch": args.batch,
        "tokens": args.tokens,
        "warmup_tokens": args.warmup_tokens,
        "gdn_layers": 24,
        "full_attention_layers": 8,
        "mode": "dense MR V2 packed decode, eager, prefix cache off, fixed length",
        "baseline": baseline,
        "profile": profile,
        "events": events,
        "derived_ms_per_decode_step": {
            "baseline_tpot": tpot,
            "gdn_module": module,
            "gdn_core": core,
            "gdn_conv": conv,
            "packed_recurrent": packed,
            "gdn_projections_norm": module - core,
            "gdn_core_other": core - conv - packed,
            "outside_gdn_module": tpot - module,
        },
        "upper_bounds": {
            "packed_recurrent_share_of_tpot": packed / tpot,
            "speedup_if_packed_recurrent_is_free": tpot / remaining,
            "speedup_if_packed_recurrent_is_5x_faster": tpot / (remaining + packed / 5),
            "note": "The packed kernel also includes gating and normalization; these are whole-kernel bounds, not isolated state-traffic savings.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    _WORKER_OUTPUT.unlink()
    print(json.dumps({"batch": args.batch, **payload["derived_ms_per_decode_step"], **payload["upper_bounds"]}))


if __name__ == "__main__":
    main()
