import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import verify_pool_production_contract as base
from verify_phase6_consistency import error, pool_call


def summary(values):
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def measure(operation, warmup, samples):
    for _ in range(warmup):
        operation()
    torch.npu.synchronize()
    npu_times = []
    host_times = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        started_ns = time.perf_counter_ns()
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        ended_ns = time.perf_counter_ns()
        npu_times.append(start.elapsed_time(end))
        host_times.append((ended_ns - started_ns) / 1e6)
    return result, {"npu_ms": summary(npu_times), "host_ms": summary(host_times)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device("npu:0")
    spec = {
        "name": "cached_contiguous",
        "lengths": [args.length],
        "has_initial": [True],
        "alias": False,
        "stride": base.ROW_SIZE,
    }
    data, _, reference_state, _, actual_state, _ = base.build_case(spec, 96, 20261109)
    data["dst"] = torch.tensor([40], dtype=torch.int32, device="npu")
    output, baseline = measure(lambda: base.production_reference(data, reference_state)[0], args.warmup, args.samples)
    actual, pool = measure(lambda: pool_call(data, actual_state, "stable"), args.warmup, args.samples)
    result = {
        "device": torch.npu.get_device_name(0),
        "stream": "accessor",
        "case": {"tokens": args.length, "source_slot": 3, "destination_slot": 40, "cache": "BF16"},
        "warmup": args.warmup,
        "samples": args.samples,
        "production_fla_with_compact_io": baseline,
        "native_pool": pool,
        "pool_output_vs_production": error(actual, output),
        "pool_state_vs_production": error(actual_state, reference_state),
        "wall_speedup": baseline["host_ms"]["median"] / pool["host_ms"]["median"],
        "npu_speedup": baseline["npu_ms"]["median"] / pool["npu_ms"]["median"],
        "library": __import__("fla_npu").__file__,
        "op_api": __import__("os").environ.get("FLA_NPU_OP_API_LIB"),
        "op_api_sha256": hashlib.sha256(Path(__import__("os").environ["FLA_NPU_OP_API_LIB"]).read_bytes()).hexdigest(),
    }
    result["passed"] = result["pool_output_vs_production"]["bitwise"] and result["pool_state_vs_production"]["bitwise"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
