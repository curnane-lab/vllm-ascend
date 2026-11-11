from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def intervals(rows, predicate):
    result = defaultdict(list)
    for row in rows:
        if not predicate(row):
            continue
        try:
            start = float(row["Start Time(us)"])
            duration = float(row["Duration(us)"])
        except (ValueError, TypeError):
            continue
        result[row["Stream ID"]].append((start, start + duration))
    return result


def union_length(intervals_by_stream):
    flattened = [interval for values in intervals_by_stream.values() for interval in values]
    merged = []
    for start, end in sorted(flattened):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged), merged


def analyze_trace(path: Path, ttft: float) -> dict:
    kernel_path = path / "ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    with kernel_path.open() as stream:
        rows = list(csv.DictReader(stream))
    all_intervals = intervals(rows, lambda row: True)
    core_intervals = intervals(rows, lambda row: "ChunkGatedDeltaRuleFwd" in row["Name"])
    device_union, merged = union_length(all_intervals)
    core_union, core_merged = union_length(core_intervals)
    op_path = path / "ASCEND_PROFILER_OUTPUT/op_statistic.csv"
    with op_path.open() as stream:
        op_rows = list(csv.DictReader(stream))
    components = [
        {
            "name": row["OP Type"],
            "count": int(row["Count"]),
            "device_time_ms": float(row["Total Time(us)"]) / 1000,
            "device_sum_percent": float(row["Ratio(%)"]),
        }
        for row in op_rows
    ]
    fraction = core_union / 1e6 / ttft
    speedups = {str(local): 1 / (1 - fraction + fraction / local) for local in (1.5, 1.72, 1.94)}
    speedups["infinite"] = 1 / (1 - fraction)
    required = None if fraction <= 1 / 11 else fraction / (fraction - 1 / 11)
    return {
        "trace_path": str(path),
        "kernel_details_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "stream_count": len(all_intervals),
        "device_union_ms": device_union / 1000,
        "core_union_ms": core_union / 1000,
        "core_interval_count": len(core_merged),
        "trace_span_ms": (merged[-1][1] - merged[0][0]) / 1000,
        "formal_ttft_geomean_seconds": ttft,
        "core_fraction_of_formal_ttft": fraction,
        "core_device_sum_percent": next(
            row["device_sum_percent"] for row in components if row["name"] == "ChunkGatedDeltaRuleFwd"
        ),
        "speedup_upper_estimates": speedups,
        "required_local_speedup_for_1p10": required,
        "free_core_has_1p10_space": speedups["infinite"] >= 1.10,
        "components": components,
        "note": "core trace and formal TTFT come from separate runs; this is a cost estimate, not a candidate speedup",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trace", action="append", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    points = []
    for length, trace_path in args.trace:
        ttft = baseline["per_length"][length]["ttft"]["geometric_mean"]
        points.append(
            {
                "prompt_tokens": int(length),
                **analyze_trace(Path(trace_path), ttft),
            }
        )
    report = {
        "schema_version": 1,
        "scope": "910B3 development cost screening only; no SparseGDN performance result",
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "formula": "S_TTFT = 1 / (1 - f + f/s)",
        "points": points,
        "native_performance_permission": (
            "at least one free-core budget point has space; this does not unlock kernel work before E1/E2"
        ),
        "910c_status": "NOT-RUN",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "length": point["prompt_tokens"],
                    "core_ms": point["core_union_ms"],
                    "f": point["core_fraction_of_formal_ttft"],
                    "required_s": point["required_local_speedup_for_1p10"],
                    "free_core_space": point["free_core_has_1p10_space"],
                    "s_1p94": point["speedup_upper_estimates"]["1.94"],
                }
                for point in points
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
