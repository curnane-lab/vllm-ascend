from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_sparsegdn_baseline import (
    make_prompts,
    read_metrics,
    stream_completion,
    wait_ready,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-port", type=int, default=18094)
    parser.add_argument("--fresh-port", type=int, default=18090)
    parser.add_argument("--model", type=Path, default=Path("/data/models/Qwen3.5-4B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3000)
    args = parser.parse_args()
    cached_ready = wait_ready(args.cached_port, args.timeout)
    fresh_ready = wait_ready(args.fresh_port, 30)
    prompt = make_prompts(args.model, [1024])[1024]
    changed_suffix = list(prompt)
    changed_suffix[960:] = changed_suffix[960:][::-1]
    runs = [
        ("cached_X_cold", args.cached_port, prompt),
        ("cached_X_warm", args.cached_port, prompt),
        ("cached_Y_shared_prefix", args.cached_port, changed_suffix),
        ("fresh_Y_no_cache", args.fresh_port, changed_suffix),
    ]
    records = []
    for name, port, token_ids in runs:
        record = stream_completion(port, token_ids, 32)
        record["name"] = name
        records.append(record)
        print(json.dumps({"completed": name, "success": record["success"]}), flush=True)
    prefix_metrics = {
        name: values for name, values in read_metrics(args.cached_port).items() if "prefix" in name.lower()
    }
    result = {
        "schema_version": 1,
        "scope": "dense cached-prefix smoke only; not SparseGDN compatibility",
        "cached_ready": cached_ready,
        "fresh_ready": fresh_ready,
        "records": records,
        "prefix_metrics": prefix_metrics,
        "cold_warm_X_text_match": records[0]["generated_text_sha256"] == records[1]["generated_text_sha256"],
        "cached_vs_fresh_Y_text_match": records[2]["generated_text_sha256"] == records[3]["generated_text_sha256"],
    }
    result["passed"] = (
        all(record["success"] for record in records)
        and result["cold_warm_X_text_match"]
        and result["cached_vs_fresh_Y_text_match"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
