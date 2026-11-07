import argparse
import gc
import json
import math
import statistics
import time

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops.gdn_pool_io import run_indexed_pool_prefill


def summarize(samples):
    ordered = sorted(samples)
    return {
        "rounds": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[math.ceil(0.95 * len(samples)) - 1],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


class PrefillCase:
    def __init__(self, args):
        self.workload = args.workload
        self.batch = args.batch
        self.length = args.suffix_length
        self.layers = args.layers
        self.src = (
            torch.full((args.batch,), 5, dtype=torch.int32, device=args.device)
            if args.workload == "cached"
            else torch.arange(17, 17 + args.batch, dtype=torch.int32, device=args.device)
        )
        self.dst = torch.arange(17, 17 + args.batch, dtype=torch.int32, device=args.device)
        self.has_initial = torch.full((args.batch,), args.workload != "fresh", dtype=torch.bool, device=args.device)
        total = args.batch * args.suffix_length
        torch.manual_seed(args.seed + args.batch * 1009 + args.suffix_length)
        self.q = torch.randn(1, total, args.key_heads, args.head_dim, dtype=torch.bfloat16, device=args.device)
        self.k = torch.randn_like(self.q)
        self.v = 0.1 * torch.randn(1, total, args.value_heads, args.head_dim, dtype=torch.bfloat16, device=args.device)
        self.g = -torch.rand(1, total, args.value_heads, dtype=torch.float32, device=args.device)
        self.beta = torch.rand(1, total, args.value_heads, dtype=torch.bfloat16, device=args.device)
        self.cu = torch.arange(0, total + 1, args.suffix_length, dtype=torch.int32, device=args.device)
        self.actual = torch.diff(self.cu)
        self.scale = args.head_dim**-0.5
        pool_rows = max(64, 18 + args.batch)
        self.initial_pool = 0.05 * torch.randn(
            args.layers,
            pool_rows,
            args.value_heads,
            args.head_dim,
            args.head_dim,
            dtype=torch.bfloat16,
            device=args.device,
        )
        self.initial_conv = 0.05 * torch.randn(
            args.layers,
            pool_rows,
            args.key_heads * 2 + args.value_heads,
            3,
            dtype=torch.bfloat16,
            device=args.device,
        )


def prepare(data, pool, conv):
    pool.copy_(data.initial_pool)
    conv.copy_(data.initial_conv)


def legacy_step(data, pool, conv):
    if data.workload == "cached":
        conv[data.dst.long()] = conv[data.src.long()]
        pool[data.dst.long()] = pool[data.src.long()]
    initial = pool[data.dst.long()].contiguous()
    initial[~data.has_initial] = 0
    query = l2norm_fwd(data.q.contiguous()).squeeze(0).contiguous()
    key = l2norm_fwd(data.k.contiguous()).squeeze(0).contiguous()
    output, final_state = torch.ops.npu.chunk_gated_delta_rule(
        query,
        key,
        data.v.squeeze(0).contiguous(),
        beta=data.beta.squeeze(0),
        initial_state=initial,
        actual_seq_lengths=data.actual,
        scale=data.scale,
        g=data.g.squeeze(0),
    )
    pool[data.dst.long()] = final_state
    return output.unsqueeze(0)


def pool_step(data, pool, conv):
    if data.workload == "cached":
        conv[data.dst.long()] = conv[data.src.long()]
    return run_indexed_pool_prefill(
        data.q,
        data.k,
        data.v,
        data.g,
        data.beta,
        pool,
        data.src,
        data.has_initial,
        data.scale,
        dst_slots=data.dst,
        cu_seqlens=data.cu,
    )


def run_layers(data, pool, conv, step):
    for layer in range(data.layers):
        output = step(data, pool[layer], conv[layer])
    return output


def peak_allocated(data, pool, conv, step):
    prepare(data, pool, conv)
    gc.collect()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    run_layers(data, pool, conv, step)
    torch.npu.synchronize()
    return {
        "incremental_peak_bytes": torch.npu.max_memory_allocated() - before,
        "retained_bytes_after_call": torch.npu.memory_allocated() - before,
    }


def main():
    """Check bitwise equality, then measure alternating legacy/pool rounds."""
    parser = argparse.ArgumentParser(description="Compare vLLM legacy GDN prefill state I/O with indexed pool I/O.")
    parser.add_argument("--workload", choices=["cached", "chunked", "fresh"], default="cached")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--suffix-length", type=int, default=128)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batch, args.suffix_length, args.layers, args.warmup) < 1 or args.rounds < 2:
        raise ValueError("batch, suffix length, layers and warmup must be positive; rounds >= 2")
    if not hasattr(torch.ops.npu, "chunk_gated_delta_rule_pool"):
        raise RuntimeError("chunk_gated_delta_rule_pool is required for this benchmark")

    torch.npu.set_device(args.device)
    data = PrefillCase(args)
    legacy_pool = data.initial_pool.clone()
    indexed_pool = data.initial_pool.clone()
    legacy_conv = data.initial_conv.clone()
    indexed_conv = data.initial_conv.clone()

    prepare(data, legacy_pool, legacy_conv)
    prepare(data, indexed_pool, indexed_conv)
    expected = run_layers(data, legacy_pool, legacy_conv, legacy_step)
    actual = run_layers(data, indexed_pool, indexed_conv, pool_step)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for layer in range(args.layers):
        torch.testing.assert_close(indexed_pool[layer], legacy_pool[layer], rtol=0, atol=0)
        torch.testing.assert_close(indexed_conv[layer], legacy_conv[layer], rtol=0, atol=0)
        if args.workload == "cached":
            torch.testing.assert_close(
                indexed_pool[layer, data.src.long()],
                data.initial_pool[layer, data.src.long()],
                rtol=0,
                atol=0,
            )

    paths = [
        ("legacy", legacy_pool, legacy_conv, legacy_step),
        ("pool", indexed_pool, indexed_conv, pool_step),
    ]
    for _ in range(args.warmup):
        for _, pool, conv, step in paths:
            run_layers(data, pool, conv, step)
    torch.npu.synchronize()
    samples = [[], []]
    for round_index in range(args.rounds):
        order = (0, 1) if round_index % 2 == 0 else (1, 0)
        for index in order:
            _, pool, conv, step = paths[index]
            prepare(data, pool, conv)
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            run_layers(data, pool, conv, step)
            torch.npu.synchronize()
            samples[index].append((time.perf_counter_ns() - start) / 1e6)
    memory = [peak_allocated(data, pool, conv, step) for _, pool, conv, step in paths]
    legacy = summarize(samples[0])
    pooled = summarize(samples[1])
    result = {
        "device": torch.npu.get_device_name(0),
        "workload": args.workload,
        "batch": args.batch,
        "suffix_tokens_per_request": args.suffix_length,
        "layers": args.layers,
        "key_heads": args.key_heads,
        "value_heads": args.value_heads,
        "head_dim": args.head_dim,
        "dtype": "bfloat16",
        "correctness": "bitwise output, complete temporal pool and convolution pool match",
        "scope": (
            f"{args.layers}-layer state I/O; cached path includes temporal CoW and both paths include convolution CoW"
        ),
        "memory_scope": "incremental per-layer peak with allocator reuse, not full-model HBM savings",
        "timing": "synchronized rounds with alternating path order",
        "legacy": legacy,
        "pool": pooled,
        "speedup": legacy["median_ms"] / pooled["median_ms"],
        "median_reduction_percent": 100 * (1 - pooled["median_ms"] / legacy["median_ms"]),
        "legacy_memory": memory[0],
        "pool_memory": memory[1],
        "incremental_peak_saved_bytes": memory[0]["incremental_peak_bytes"] - memory[1]["incremental_peak_bytes"],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        f"{args.workload} B={args.batch} L={args.suffix_length} layers={args.layers}: "
        f"legacy={legacy['median_ms']:.4f} ms, pool={pooled['median_ms']:.4f} ms, "
        f"speedup={result['speedup']:.3f}x, "
        f"peak saved={result['incremental_peak_saved_bytes'] / 1024**2:.2f} MiB"
    )
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
