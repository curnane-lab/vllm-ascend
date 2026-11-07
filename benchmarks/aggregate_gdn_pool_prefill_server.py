import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values) if values else None


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def requests_by_round(result):
    for batch in result["rounds"]:
        yield batch["wall_ms"], batch["output_tokens_per_second"], batch["requests"]


def main():
    """Aggregate wall, throughput, TTFT, and output-id parity by trial."""
    parser = argparse.ArgumentParser(description="Aggregate paired GDN server A/B result files.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.input]
    grouped = defaultdict(dict)
    for result in results:
        key = (result["server"]["chunked_prefill_size"], result["trial"])
        if result["mode"] in grouped[key]:
            raise ValueError(f"duplicate {result['mode']} result for {key}")
        grouped[key][result["mode"]] = result

    configurations = []
    total_matched = 0
    total_pairs = 0
    for (chunk_size, trial), pair in sorted(grouped.items()):
        if set(pair) != {"legacy", "pool"}:
            raise ValueError(f"need legacy and pool results for chunk={chunk_size}, trial={trial}")
        legacy = pair["legacy"]
        pool = pair["pool"]
        for field in ("model", "server", "workload"):
            if legacy[field] != pool[field]:
                raise ValueError(f"{field} mismatch for chunk={chunk_size}, trial={trial}")
        legacy_rounds = list(requests_by_round(legacy))
        pool_rounds = list(requests_by_round(pool))
        if len(legacy_rounds) != len(pool_rounds):
            raise ValueError(f"round count mismatch for chunk={chunk_size}, trial={trial}")

        matched = 0
        pairs = 0
        for (_, _, legacy_requests), (_, _, pool_requests) in zip(legacy_rounds, pool_rounds):
            if len(legacy_requests) != len(pool_requests):
                raise ValueError(f"request count mismatch for chunk={chunk_size}, trial={trial}")
            for legacy_request, pool_request in zip(legacy_requests, pool_requests):
                for field in ("rid", "input_tokens", "cached_tokens", "decode_tokens"):
                    if legacy_request[field] != pool_request[field]:
                        raise ValueError(f"{field} mismatch for chunk={chunk_size}, trial={trial}")
                pairs += 1
                if legacy_request["output_ids"] == pool_request["output_ids"]:
                    matched += 1
        total_matched += matched
        total_pairs += pairs

        legacy_walls = [wall for wall, _, _ in legacy_rounds]
        pool_walls = [wall for wall, _, _ in pool_rounds]
        legacy_throughput = [throughput for _, throughput, _ in legacy_rounds]
        pool_throughput = [throughput for _, throughput, _ in pool_rounds]
        legacy_ttft = [
            request["ttft_ms"]
            for _, _, requests in legacy_rounds
            for request in requests
            if request["ttft_ms"] is not None
        ]
        pool_ttft = [
            request["ttft_ms"]
            for _, _, requests in pool_rounds
            for request in requests
            if request["ttft_ms"] is not None
        ]
        legacy_wall_median = median(legacy_walls)
        pool_wall_median = median(pool_walls)
        legacy_ttft_median = median(legacy_ttft)
        pool_ttft_median = median(pool_ttft)
        configurations.append(
            {
                "chunked_prefill_size": chunk_size,
                "trial": trial,
                "primary_prefill_shape": (
                    f"B{chunk_size // legacy['workload']['suffix_tokens_per_request']} "
                    f"x {legacy['workload']['suffix_tokens_per_request']}"
                ),
                "legacy_device": legacy["device"],
                "pool_device": pool["device"],
                "wall_samples_ms": {"legacy": legacy_walls, "pool": pool_walls},
                "throughput_samples_tokens_per_second": {
                    "legacy": legacy_throughput,
                    "pool": pool_throughput,
                },
                "legacy_wall_ms": legacy_wall_median,
                "pool_wall_ms": pool_wall_median,
                "wall_speedup": ratio(legacy_wall_median, pool_wall_median),
                "legacy_throughput_tokens_per_second": median(legacy_throughput),
                "pool_throughput_tokens_per_second": median(pool_throughput),
                "throughput_speedup": ratio(median(pool_throughput), median(legacy_throughput)),
                "legacy_ttft_ms": legacy_ttft_median,
                "pool_ttft_ms": pool_ttft_median,
                "ttft_speedup": ratio(legacy_ttft_median, pool_ttft_median),
                "matched_request_pairs": matched,
                "total_request_pairs": pairs,
                "legacy_instrumentation": legacy["instrumentation"],
                "pool_instrumentation": pool["instrumentation"],
            }
        )

    by_chunk = defaultdict(list)
    for config in configurations:
        by_chunk[config["chunked_prefill_size"]].append(config)
    summaries = []
    for chunk_size, configs in sorted(by_chunk.items()):
        legacy_walls = []
        pool_walls = []
        legacy_throughputs = []
        pool_throughputs = []
        legacy_ttft_samples = []
        pool_ttft_samples = []
        for (trial_chunk, _), pair in sorted(grouped.items()):
            if trial_chunk != chunk_size:
                continue
            for result, walls, throughputs, ttft_samples in (
                (pair["legacy"], legacy_walls, legacy_throughputs, legacy_ttft_samples),
                (pair["pool"], pool_walls, pool_throughputs, pool_ttft_samples),
            ):
                for wall, throughput, requests in requests_by_round(result):
                    walls.append(wall)
                    throughputs.append(throughput)
                    ttft_samples.extend(request["ttft_ms"] for request in requests if request["ttft_ms"] is not None)
        legacy_wall = median(legacy_walls)
        pool_wall = median(pool_walls)
        legacy_throughput = median(legacy_throughputs)
        pool_throughput = median(pool_throughputs)
        legacy_ttft = median(legacy_ttft_samples)
        pool_ttft = median(pool_ttft_samples)
        summaries.append(
            {
                "chunked_prefill_size": chunk_size,
                "trials": len(configs),
                "measured_rounds_per_arm": len(legacy_walls),
                "legacy_wall_ms": legacy_wall,
                "pool_wall_ms": pool_wall,
                "wall_speedup": ratio(legacy_wall, pool_wall),
                "legacy_throughput_tokens_per_second": legacy_throughput,
                "pool_throughput_tokens_per_second": pool_throughput,
                "throughput_speedup": ratio(pool_throughput, legacy_throughput),
                "legacy_ttft_ms": legacy_ttft,
                "pool_ttft_ms": pool_ttft,
                "ttft_speedup": ratio(legacy_ttft, pool_ttft),
                "matched_request_pairs": sum(config["matched_request_pairs"] for config in configs),
                "total_request_pairs": sum(config["total_request_pairs"] for config in configs),
            }
        )

    output = {
        "workload": results[0]["workload"],
        "server": {key: value for key, value in results[0]["server"].items() if key != "chunked_prefill_size"},
        "chunked_prefill_sizes": sorted({chunk_size for chunk_size, _ in grouped}),
        "model": results[0]["model"],
        "aggregation": {
            "wall_and_throughput": "median of all per-round values across trials",
            "ttft": "absolute median across all requests and trials",
            "no_outlier_rounds_removed": True,
        },
        "matched_request_pairs": total_matched,
        "total_request_pairs": total_pairs,
        "summaries": summaries,
        "configurations": configurations,
        "inputs": args.input,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"matched {total_matched}/{total_pairs} request pairs; results written to {output_path}")
    for summary in summaries:
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
