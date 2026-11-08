#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "$script_dir/../.." && pwd)
runtime=${GDN_PHASE6_RUNTIME:?set GDN_PHASE6_RUNTIME to the installed native runtime}
vllm_source=${VLLM_SOURCE_DIR:?set VLLM_SOURCE_DIR to the matching vLLM checkout}
result_dir=${GDN_RESULTS_DIR:-$repo_dir/benchmarks/results/gdn_pool_native}
device=${GDN_PHASE6_DEVICE:-3}
port=${GDN_PHASE6_PORT:-39100}
trials=${TRIALS:-5}
rounds=${ROUNDS:-5}
output_tokens=${OUTPUT_TOKENS:-16}
label=${LABEL:-phase6_native}
scenarios=${SCENARIOS:-"eager_c1 eager_c64 graph_c1 graph_c64"}

source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash}"
export GDN_PHASE6_RUNTIME="$runtime"
export GDN_RESULTS_DIR="$result_dir"
export VLLM_ASCEND_GDN_POOL_LAYOUT_FUSION=0
export PYTHONPATH="$runtime:$script_dir:$repo_dir:$vllm_source:${PYTHONPATH:-}"
mkdir -p "$result_dir"

run_arm() {
    local scenario=$1
    local direction=$2
    local trial=$3
    local mode=$4
    local clients=$5
    local graph_mode=$6
    local output="$result_dir/${label}_${scenario}_${direction}_t${trial}_${mode}.json"
    python "$script_dir/run_server_case.py" \
        --mode "$mode" \
        --chunk-size 512 \
        --trial "$trial" \
        --device "$device" \
        --port "$port" \
        --clients "$clients" \
        --rounds "$rounds" \
        --output-tokens "$output_tokens" \
        --graph-mode "$graph_mode" \
        --startup-timeout 1200 \
        --no-instrument \
        --output "$output"
    port=$((port + 1))
}

for scenario in $scenarios; do
    case "$scenario" in
        eager_c1)
            clients=1
            graph_mode=NONE
            scenario_offset=0
            ;;
        eager_c64)
            clients=64
            graph_mode=NONE
            scenario_offset=100
            ;;
        graph_c1)
            clients=1
            graph_mode=FULL_AND_PIECEWISE
            scenario_offset=200
            ;;
        graph_c64)
            clients=64
            graph_mode=FULL_AND_PIECEWISE
            scenario_offset=300
            ;;
        *)
            exit 2
            ;;
    esac
    overall_inputs=()
    for direction in forward reverse; do
        direction_inputs=()
        for ((index = 0; index < trials; ++index)); do
            trial=$((scenario_offset + index))
            if [[ "$direction" == reverse ]]; then
                trial=$((trial + trials))
                arms=(pool legacy)
            else
                arms=(legacy pool)
            fi
            for mode in "${arms[@]}"; do
                run_arm "$scenario" "$direction" "$trial" "$mode" "$clients" "$graph_mode"
            done
            for mode in legacy pool; do
                input="$result_dir/${label}_${scenario}_${direction}_t${trial}_${mode}.json"
                direction_inputs+=("$input")
                overall_inputs+=("$input")
            done
        done
        python "$repo_dir/benchmarks/aggregate_gdn_pool_prefill_server.py" \
            --input "${direction_inputs[@]}" \
            --output "$result_dir/${label}_${scenario}_${direction}_ab.json"
    done
    python "$repo_dir/benchmarks/aggregate_gdn_pool_prefill_server.py" \
        --input "${overall_inputs[@]}" \
        --output "$result_dir/${label}_${scenario}_ab.json"
    python - "$result_dir" "$label" "$scenario" "${overall_inputs[@]}" <<'PY'
import json
from pathlib import Path
import sys

result_dir, label, scenario, *inputs = sys.argv[1:]
result_dir = Path(result_dir)
raw = [json.loads(Path(path).read_text()) for path in inputs]
results = {}
for direction in ("forward", "reverse"):
    path = result_dir / f"{label}_{scenario}_{direction}_ab.json"
    results[direction] = json.loads(path.read_text())
overall = json.loads((result_dir / f"{label}_{scenario}_ab.json").read_text())
metadata = [result["phase6_pool"] for result in raw]
first = metadata[0]
for item in metadata[1:]:
    for field in (
        "stable_stream", "stable_library_sha256", "op_api_sha256",
        "pool_op_api_sha256", "kernel_sha256", "full_op_api_sha256",
        "vllm_op_api_sha256", "layout_fusion", "cow_elision", "git_head",
        "sync_scheduling", "prefix_cache", "fla_backend",
    ):
        if item[field] != first[field]:
            raise RuntimeError(f"runtime mismatch: {field}")
passed = all(result["matched_request_pairs"] == result["total_request_pairs"] for result in results.values())
passed = passed and overall["matched_request_pairs"] == overall["total_request_pairs"]
passed = passed and all(result["phase6_pool"]["native_operator_observed"] for result in raw if result["mode"] == "pool")
passed = passed and all(
    result["server"]["acl_graph_replay_observed"]
    for result in raw
    if result["server"]["requested_cudagraph_mode"] != "NONE"
)
combined = {
    "scenario": scenario,
    "passed": passed,
    "protocol": {
        "paired_trials_per_direction": len(inputs) // 4,
        "launch_directions": ("legacy,pool", "pool,legacy"),
        "all_rounds_retained": True,
        "median_reported_per_direction_and_overall": True,
        "graph_replay_required": True,
    },
    "runtime": first,
    "directions": results,
    "overall": overall,
    "inputs": inputs,
}
output = result_dir / f"{label}_{scenario}_verified_ab.json"
output.write_text(json.dumps(combined, indent=2) + "\n")
print(json.dumps({
    "scenario": scenario,
    "passed": passed,
    "matched": overall["matched_request_pairs"],
    "pairs": overall["total_request_pairs"],
    "output": str(output),
}, indent=2))
if not passed:
    raise SystemExit(1)
PY
done
