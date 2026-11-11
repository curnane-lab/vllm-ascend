from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_sparsegdn_baseline import make_prompts, stream_completion, wait_ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=Path, default=Path("/data/models/Qwen3.5-4B"))
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--suffix", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--only", choices=("independent_P", "P_plus_X", "P_plus_Y"))
    parser.add_argument("--request-index", type=int, default=0)
    args = parser.parse_args()
    ready = wait_ready(args.port, args.timeout)
    if not ready["ready"]:
        args.output.write_text(json.dumps(ready, indent=2) + "\n")
        raise SystemExit(1)
    total = args.prefix + args.suffix
    full = make_prompts(args.model, [total])[total]
    prompt_y = list(full)
    prompt_y[args.prefix :] = prompt_y[args.prefix :][::-1]
    requests = [
        ("independent_P", full[: args.prefix]),
        ("P_plus_X", full),
        ("P_plus_Y", prompt_y),
    ]
    if args.only:
        requests = [item for item in requests if item[0] == args.only]
    records = []
    for index, (name, prompt) in enumerate(requests):
        request_index = args.request_index if args.only else index
        record = stream_completion(args.port, prompt, 1)
        record.update(
            {
                "name": name,
                "request_index": request_index,
                "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
                "scope": "SCOPE-A cold full-prompt FP32 reference; not a performance or quality result",
            }
        )
        records.append(record)
        print(json.dumps({"completed": name, "success": record["success"]}), flush=True)
        if not record["success"]:
            break
    result = {
        "schema_version": 1,
        "prefix_tokens": args.prefix,
        "suffix_tokens": args.suffix,
        "request_order": [name for name, _ in requests],
        "records": records,
        "passed": len(records) == len(requests) and all(record["success"] for record in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
