import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

PATTERN = ["gdn", "gdn", "gdn", "full_attention"] * 8


def _busy(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _is_gate(row):
    return row["OP Type"].startswith("MatMul") and row["shape"].endswith(";18432,2560")


def _is_down(row):
    return row["OP Type"].startswith("MatMul") and row["shape"].endswith(";2560,9216")


def _is_other(row):
    return row["OP Type"] == "ArgMaxV2" or row["shape"].endswith(";248320,2560")


def _load_layers(rows):
    layers = []
    for gate, row in enumerate(rows):
        if not _is_gate(row):
            continue
        start = next((i for i in range(gate - 1, -1, -1) if _is_down(rows[i])), -1) + 1
        down = next(i for i in range(gate + 1, len(rows)) if _is_down(rows[i]))
        names = {rows[i]["OP Type"] for i in range(start, down + 1)}
        if "d_recurrent_gated_delta_rule_packed_decode_kernel" in names:
            kind = "gdn"
        elif "FusedInferAttentionScore" in names:
            kind = "full_attention"
        else:
            kind = "other"
        layers.append({"kind": kind, "start": start, "gate": gate, "end": down + 1})
    return layers


def _is_capture_only(row, kind):
    if row["OP Type"] in {"Conv2D", "TransData", "MemSet", "StridedSlice", "Swish", "Transpose"}:
        return True
    return kind == "gdn" and row["OP Type"] == "Slice" and row["duration"] > 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    summaries = sorted(args.trace_dir.glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
    if len(summaries) != 1:
        raise RuntimeError(f"expected one successful op_summary, found {len(summaries)}")
    summary = summaries[0]

    rows = list(csv.DictReader(summary.open()))
    for row in rows:
        row["start"] = float(row["Task Start Time(us)"])
        row["duration"] = float(row["Task Duration(us)"])
        row["end"] = row["start"] + row["duration"]
        row["shape"] = row["Input Shapes"].strip('"')
    rows.sort(key=lambda row: row["start"])
    layers = _load_layers(rows)
    steps = []
    index = 0
    while index <= len(layers) - 32:
        window = layers[index : index + 32]
        if [layer["kind"] for layer in window] == PATTERN:
            steps.append(window)
            index += 32
        else:
            index += 1
    steps = [
        window
        for window in steps
        if any(
            _is_other(rows[index])
            for layer in window
            for index in range(layer["start"], layer["end"])
        )
    ]
    steps = steps[-metrics["decode_steps"] :]
    if len(steps) != metrics["decode_steps"]:
        raise RuntimeError(f"expected {metrics['decode_steps']} complete steps, found {len(steps)}")

    step_records = []
    for window in steps:
        categories = {name: [] for name in ("gdn", "full_attention", "mlp", "other")}
        op_durations = {name: collections.Counter() for name in categories}
        for layer in window:
            kind = layer["kind"]
            for index in range(layer["start"], layer["gate"]):
                if _is_capture_only(rows[index], kind):
                    continue
                target = "other" if _is_other(rows[index]) else kind
                categories[target].append((rows[index]["start"], rows[index]["end"]))
                op_durations[target][rows[index]["OP Type"]] += rows[index]["duration"]
            for index in range(layer["gate"], layer["end"]):
                target = "other" if _is_other(rows[index]) else "mlp"
                categories[target].append((rows[index]["start"], rows[index]["end"]))
                op_durations[target][rows[index]["OP Type"]] += rows[index]["duration"]
        device = {name: _busy(intervals) for name, intervals in categories.items()}
        all_intervals = [interval for intervals in categories.values() for interval in intervals]
        step_records.append(
            {
                "device_ms": {name: value / 1e3 for name, value in device.items()},
                "device_union_ms": _busy(all_intervals) / 1e3,
                "op_duration_ms": {
                    name: {op: value / 1e3 for op, value in counter.items()}
                    for name, counter in op_durations.items()
                },
            }
        )

    # The final step has no following step boundary for output/sampling attribution.
    step_records = step_records[:-1]
    names = ("gdn", "full_attention", "mlp", "other")
    means = {
        name: statistics.mean(record["device_ms"][name] for record in step_records)
        for name in names
    }
    device_union = statistics.mean(record["device_union_ms"] for record in step_records)
    tpot = metrics["baseline"]["tpot_ms"]["mean"]
    host = tpot - device_union
    overlap = device_union - sum(means.values())
    if device_union > tpot * 1.05 or host < -tpot * 0.05 or abs(overlap) > tpot * 0.05:
        raise RuntimeError(
            f"invalid replay account: tpot={tpot:.3f} device={device_union:.3f} "
            f"host={host:.3f} overlap={overlap:.3f}"
        )
    op_means = {name: collections.Counter() for name in names}
    for record in step_records:
        for name in names:
            for op, value in record["op_duration_ms"][name].items():
                op_means[name][op] += value
    op_means = {
        name: {op: value / len(step_records) for op, value in counter.most_common()}
        for name, counter in op_means.items()
    }
    payload = {
        "schema_version": 1,
        "batch": metrics["batch"],
        "mode": metrics["mode"],
        "metrics": metrics,
        "source": str(summary),
        "steps": {
            "complete_pattern_matches": len(steps),
            "used": len(step_records),
            "discarded_last": 1,
            "layers_per_step": 32,
        },
        "mean_ms_per_decode_step": {
            "baseline_tpot": tpot,
            "device_union": device_union,
            "host_estimate": host,
            **means,
            "outside_gdn": host + means["full_attention"] + means["mlp"] + means["other"],
            "category_sum": sum(means.values()),
            "category_overlap_adjustment": device_union - sum(means.values()),
        },
        "op_duration_ms_per_decode_step": op_means,
        "step_records": step_records,
        "notes": [
            "Device steps use non-overlapping 32-layer patterns and must contain vocab projection or ArgMax.",
            "Known capture/prepare kernels and long GDN-only slices are excluded from replay accounting.",
            "host_estimate is baseline TPOT minus device union, not a per-step host trace measurement.",
            "Category device times use interval unions; op durations explain composition and need not sum to the unions.",
            "The final complete step is discarded because it lacks a following step boundary.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"batch": metrics["batch"], **payload["mean_ms_per_decode_step"]}))


if __name__ == "__main__":
    main()
