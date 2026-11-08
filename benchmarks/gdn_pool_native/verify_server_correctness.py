import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for execution in ("eager", "graph"):
        for mode in ("legacy", "pool"):
            parser.add_argument(f"--{execution}-{mode}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for execution in ("eager", "graph"):
        legacy = json.loads(getattr(args, f"{execution}_legacy").read_text())
        pool = json.loads(getattr(args, f"{execution}_pool").read_text())
        pairs = 0
        matched = 0
        for legacy_round, pool_round in zip(legacy["rounds"], pool["rounds"], strict=True):
            for legacy_request, pool_request in zip(legacy_round["requests"], pool_round["requests"], strict=True):
                pairs += 1
                matched += legacy_request["output_ids"] == pool_request["output_ids"]
        graph_valid = True
        if execution == "graph":
            graph_valid = all(
                result["server"]["graph_capture_finished"] and result["server"]["acl_graph_replay_observed"]
                for result in (legacy, pool)
            )
        results[execution] = {
            "matched": matched,
            "pairs": pairs,
            "graph_replay_valid": graph_valid,
            "legacy_graph": legacy["server"],
            "pool_graph": pool["server"],
            "legacy_runtime": legacy["phase6_pool"],
            "pool_runtime": pool["phase6_pool"],
        }
    results["passed"] = all(
        item["matched"] == item["pairs"] and item["graph_replay_valid"] for item in results.values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
