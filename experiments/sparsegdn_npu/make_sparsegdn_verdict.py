from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def selector_stats(trace_dir: Path) -> list[dict]:
    result = []
    for path in sorted(trace_dir.glob("summary_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        possible = sum((row["prompt_tokens"] // 64) * 16 for row in rows)
        events = sum(row["valid_events"] for row in rows)
        result.append(
            {
                "file": path.name,
                "layers": len(rows),
                "valid_events": events,
                "capacity": sum(row["capacity"] for row in rows),
                "possible_events": possible,
                "keep_rate": events / possible,
                "all_finite": all(row["finite"] for row in rows),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace-4k", type=Path, required=True)
    parser.add_argument("--trace-8k", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    e0 = root / "E0"
    e1 = root / "E1"
    manifest = load(e0 / "manifest.json")
    baseline = load(e0 / "production_baseline.json")
    baseline_32k = load(e0 / "production_baseline_32k.json")
    ledger = load(e0 / "prefill_cost_ledger.json")
    dense = load(e1 / "dense_identity.json")
    approximation = load(e1 / "approximation_cases.json")
    vectorized = load(e1 / "vectorized_reference.json")
    partition = load(e1 / "prefix_partition_matrix.json")
    model_4k = load(e1 / "model_scopea_4k.json")
    model_8k = load(e1 / "model_scopea_8k.json")
    service_32 = load(e0 / "service_b32.json")
    service_64 = load(e0 / "service_b64.json")
    decode_64 = load(e0 / "decode_b64_negative_control.json")
    cache = load(e0 / "dense_prefix_cache_smoke.json")
    dense_cases = dense["cases"] + [dense["real_geometry_case"]]
    baseline_points = {
        length: {
            "ttft_geomean_seconds": values["ttft"]["geometric_mean"],
            "ttft_95_interval": values["ttft"].get("log_normal_95_interval"),
            "wall_geomean_seconds": values["wall"]["geometric_mean"],
            "output_tokens_per_second_wall": values["output_tokens_per_second_wall"]["geometric_mean"],
            "decode_output_tokens_per_second": values["decode_output_tokens_per_second"]["geometric_mean"],
        }
        for length, values in baseline["per_length"].items()
    }
    baseline_points["32768"] = {
        "ttft_geomean_seconds": baseline_32k["per_length"]["32768"]["ttft"]["geometric_mean"],
        "ttft_95_interval": baseline_32k["per_length"]["32768"]["ttft"].get("log_normal_95_interval"),
        "wall_geomean_seconds": baseline_32k["per_length"]["32768"]["wall"]["geometric_mean"],
        "output_tokens_per_second_wall": baseline_32k["per_length"]["32768"]["output_tokens_per_second_wall"][
            "geometric_mean"
        ],
        "decode_output_tokens_per_second": baseline_32k["per_length"]["32768"]["decode_output_tokens_per_second"][
            "geometric_mean"
        ],
    }
    budget_points = [
        {
            "prompt_tokens": point["prompt_tokens"],
            "core_union_ms": point["core_union_ms"],
            "fraction_of_ttft": point["core_fraction_of_formal_ttft"],
            "required_local_speedup_for_1p10": point["required_local_speedup_for_1p10"],
            "estimated_ttft_speedup_at_1p94": point["speedup_upper_estimates"]["1.94"],
            "estimated_ttft_speedup_with_free_core": point["speedup_upper_estimates"]["infinite"],
        }
        for point in ledger["points"]
    ]
    model_summaries = {}
    for name, model, trace in (
        ("4K", model_4k, args.trace_4k),
        ("8K", model_8k, args.trace_8k),
    ):
        model_summaries[name] = {
            "all_finite": model["summary"]["all_finite"],
            "prefix_mask_difference_layers": model["summary"]["prefix_mask_difference_cases"],
            "prefix_state_difference_layers": model["summary"]["prefix_state_difference_cases"],
            "prefix_output_difference_layers": model["summary"]["prefix_output_difference_cases"],
            "logits_XY": model["summary"]["logits_XY"],
            "selector": selector_stats(trace),
        }
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "E0+E1 first-round verdict",
        "overall_verdict": "STOP-BUDGET-PAPER-RANGE + BLOCKED-PREFIX/CHUNK-SEMANTICS",
        "hardware": "Ascend 910B3 64GB; 910C NOT-RUN",
        "model": "Qwen3.5-4B actual local weights/tokenizer; Qwen3.5-9B NOT-RUN",
        "tp": 1,
        "source_shas": {
            "vllm": manifest["sources"]["production_vllm"]["sha"],
            "vllm_ascend": manifest["sources"]["production_ascend"]["sha"],
            "experiment_worktree": manifest["sources"]["experiment_worktree"]["sha"],
        },
        "policy_version": "SparseGDN-v1-independent-reference+deferred-W-and-mask-overflow-regressions",
        "scope": {
            "A": "PASS-SCOPED-RESEARCH-ONLY: zero-state cold full-prompt prefill followed by normal dense decode",
            "B": "BLOCKED-PREFIX/CHUNK-SEMANTICS",
            "deployment_PR": "NOT-ALLOWED; no quality or 910C/e2e evidence",
        },
        "chunk_size": 64,
        "state_dtype": {
            "reference": "CPU FP64",
            "actual_model_reference": "CPU FP32",
            "production_baseline": "BF16 QKV; model-configured FP32 recurrent state",
        },
        "measured_range": {
            "dense_B1": "4K/8K/16K/32K, output=128, five blocks each",
            "dense_service": "B32/B64 mixed lengths 512/1024/2048/4096, output=128",
            "dense_negative_control": "B64 prompt=8, output=128",
            "reference": "1-5 chunks, tails, P=64/65/127/128/129/192/256/1024, K=V=8 and 128",
            "actual_model": "4K and 8K P/P+X/P+Y, suffix=128, output=1 for semantic capture only",
        },
        "actual_graph_mode": {
            "dense_decode": "FULL_DECODE_ONLY; seven graphs actually captured",
            "dense_prefill": "chunked eager trace, not FULL replay",
            "sparse_reference": "enforce-eager SCOPE-A only; not a production graph result",
        },
        "comparison_groups": {
            "D0": "current production dense measured",
            "R0_R1_A1_A2_S_ref": "independent reference measured",
            "S_npu": "NOT-RUN",
        },
        "quality": {
            "implementation_reference": "PASS after registered numerical regressions",
            "dense_identity_max_output_relF": max(
                case["R1_vs_R0"]["output"]["relative_frobenius"] for case in dense_cases
            ),
            "dense_identity_max_state_relF": max(
                case["R1_vs_R0"]["final_state"]["relative_frobenius"] for case in dense_cases
            ),
            "approximation_cases_passed": approximation["passed"],
            "vectorized_reference_passed": vectorized["passed"],
            "S_npu_implementation": "NOT-RUN",
            "E2_task_quality_NLL_generation": "NOT-RUN",
            "bit_identical_to_dense": False,
        },
        "semantics": {
            "prefix_partition": partition["summary"],
            "actual_model": model_summaries,
            "dense_prefix_cache_smoke": {
                "passed": cache["passed"],
                "cold_warm_X_text_match": cache["cold_warm_X_text_match"],
                "cached_vs_fresh_Y_text_match": cache["cached_vs_fresh_Y_text_match"],
                "prefix_metrics": cache["prefix_metrics"],
                "note": "dense cache correctness does not unlock SparseGDN scope B",
            },
        },
        "performance": {
            "candidate_speedup": "NOT-RUN; no S-npu or service candidate",
            "dense_baseline": baseline_points,
            "cost_screening": budget_points,
            "paper_range_budget_insufficient": all(
                point["estimated_ttft_speedup_at_1p94"] < 1.10 for point in budget_points
            ),
            "free_core_formal_stop_triggered": any(
                point["estimated_ttft_speedup_with_free_core"] < 1.10 for point in budget_points
            ),
            "B32_diagnostic": {
                "wall_seconds": service_32["wall_seconds"],
                "completed_output_tokens_per_second": service_32["completed_output_tokens_per_second"],
                "success_count": service_32["success_count"],
            },
            "B64_diagnostic": {
                "wall_seconds": service_64["wall_seconds"],
                "completed_output_tokens_per_second": service_64["completed_output_tokens_per_second"],
                "success_count": service_64["success_count"],
            },
            "B64_decode_negative_control": {
                "wall_seconds": decode_64["wall_seconds"],
                "completed_output_tokens_per_second": decode_64["completed_output_tokens_per_second"],
                "success_count": decode_64["success_count"],
            },
            "note": "B32/B64 are wiring diagnostics without preregistered SLO; no goodput claim",
        },
        "next_allowed_stage": {
            "native_NPU_kernel": "NOT-ALLOWED",
            "E2": (
                "NOT-RUN; only an explicitly research-scoped continuation could "
                "evaluate quality, not unlock performance delivery"
            ),
            "reason": (
                "paper-range local speedups do not meet the TTFT budget, "
                "and ordinary prefix/chunk semantics are blocked"
            ),
        },
        "not_run": [
            "910C acceptance and deployment TP/EP",
            "Qwen3.5-9B",
            "E2 task quality/NLL/long generation",
            "E3 NPU kernel and implementation correctness",
            "E4 core paired performance",
            "E5/E6 TTFT/e2e/cache/spec compatibility acceptance",
            "64K/128K and 32K profiler trace",
        ],
        "raw_artifacts": {
            "E0": str(e0),
            "E1": str(e1),
            "minimal_counterexample": str(e1 / "prefix_counterexample.pt"),
            "invalid_inputs_for_audit": "/tmp/sparsegdn_debug_trace",
            "numerical_invalid_before_fix": str(e1 / "raw/invalid_before_w_fix"),
        },
        "wall_time": {
            "manifest_timestamp_utc": manifest["timestamp_utc"],
            "verdict_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "note": "task logs preserve per-service timing; no unlogged card-hour estimate is presented as exact",
        },
    }
    output = root / "verdict.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "overall": report["overall_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
