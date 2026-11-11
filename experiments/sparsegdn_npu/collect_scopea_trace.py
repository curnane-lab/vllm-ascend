from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from sparsegdn_reference import error_metrics


def layer_files(trace_dir: Path, run_tag: str, request: int) -> dict[int, Path]:
    prefix = f"layer_{run_tag}_request{request}_language_model.model.layers."
    result = {}
    for path in trace_dir.glob(f"{prefix}*.linear_attn.pt"):
        match = re.search(r"layers\.(\d+)\.linear_attn", path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def load_logits(trace_dir: Path, run_tag: str, request: int) -> tuple[torch.Tensor, Path]:
    matches = list(trace_dir.glob(f"logits_{run_tag}_request{request}_*.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one logits file for request {request}, got {matches}")
    return torch.load(matches[0], weights_only=False).flatten().float(), matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--prefix", type=int, required=True)
    parser.add_argument("--suffix", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [layer_files(args.trace_dir, args.run_tag, request) for request in range(3)]
    expected_layers = set(range(32)) - {3, 7, 11, 15, 19, 23, 27, 31}
    if any(set(files) != expected_layers for files in paths):
        raise RuntimeError(f"incomplete layer traces: {[sorted(files) for files in paths]}")
    layers = []
    for layer in sorted(expected_layers):
        records = [torch.load(paths[request][layer], weights_only=False) for request in range(3)]
        independent, full_x, full_y = records
        prefix_chunks = args.prefix // 64
        record = {
            "layer": layer,
            "mask_differences": {
                "P_vs_PX": int((independent["mask"] != full_x["mask"][:, :prefix_chunks]).sum()),
                "P_vs_PY": int((independent["mask"] != full_y["mask"][:, :prefix_chunks]).sum()),
                "PX_vs_PY": int((full_x["mask"][:, :prefix_chunks] != full_y["mask"][:, :prefix_chunks]).sum()),
            },
            "tiers": [record["tiers"].tolist() for record in records],
            "valid_events": [int(record["mask"].sum()) for record in records],
            "U_prefix": {
                "P_vs_PX": error_metrics(independent["U"].float(), full_x["U"][:prefix_chunks].float()),
                "P_vs_PY": error_metrics(independent["U"].float(), full_y["U"][:prefix_chunks].float()),
            },
            "output_prefix": {
                "P_vs_PX": error_metrics(independent["output"].float(), full_x["output"][: args.prefix].float()),
                "P_vs_PY": error_metrics(independent["output"].float(), full_y["output"][: args.prefix].float()),
                "PX_vs_PY": error_metrics(
                    full_x["output"][: args.prefix].float(), full_y["output"][: args.prefix].float()
                ),
            },
            "state_at_prefix": {
                "P_vs_PX": error_metrics(independent["final_state"].float(), full_x["state_at_prefix"].float()),
                "P_vs_PY": error_metrics(independent["final_state"].float(), full_y["state_at_prefix"].float()),
                "PX_vs_PY": error_metrics(full_x["state_at_prefix"].float(), full_y["state_at_prefix"].float()),
            },
            "final_state_XY": error_metrics(full_x["final_state"].float(), full_y["final_state"].float()),
            "finite": all(
                bool(record["U"].float().isfinite().all())
                and bool(record["output"].float().isfinite().all())
                and bool(record["final_state"].float().isfinite().all())
                for record in records
            ),
            "raw_files": [
                {
                    "path": str(paths[request][layer]),
                    "bytes": paths[request][layer].stat().st_size,
                    "sha256": hashlib.sha256(paths[request][layer].read_bytes()).hexdigest(),
                }
                for request in range(3)
            ],
        }
        layers.append(record)
    logits = [load_logits(args.trace_dir, args.run_tag, request) for request in range(3)]
    report = {
        "schema_version": 1,
        "scope": "SCOPE-A actual Qwen3.5-4B FP32 SparseGDN reference; eager, TP1, no prefix cache",
        "compute_dtype": "float32",
        "trace_storage_dtype": {
            "U": "bfloat16",
            "output": "bfloat16",
            "states": "float32",
            "logits": "float32",
        },
        "prefix_tokens": args.prefix,
        "suffix_tokens": args.suffix,
        "request_order": ["independent_P", "P_plus_X", "P_plus_Y"],
        "run_tag": args.run_tag,
        "layers": layers,
        "logits": {
            "shapes": [list(item[0].shape) for item in logits],
            "PX_vs_PY": error_metrics(logits[1][0], logits[2][0]),
            "P_vs_PX_different_boundaries": error_metrics(logits[0][0], logits[1][0]),
            "P_vs_PY_different_boundaries": error_metrics(logits[0][0], logits[2][0]),
            "note": "P logits are after P; PX/PY logits are after P+suffix. X/Y share the final boundary.",
            "files": [
                {
                    "path": str(item[1]),
                    "sha256": hashlib.sha256(item[1].read_bytes()).hexdigest(),
                }
                for item in logits
            ],
        },
        "summary": {
            "all_finite": all(layer["finite"] for layer in layers),
            "prefix_mask_difference_cases": sum(layer["mask_differences"]["PX_vs_PY"] > 0 for layer in layers),
            "prefix_state_difference_cases": sum(
                layer["state_at_prefix"]["PX_vs_PY"]["relative_frobenius"] > 1e-6 for layer in layers
            ),
            "prefix_output_difference_cases": sum(
                layer["output_prefix"]["PX_vs_PY"]["relative_frobenius"] > 1e-6 for layer in layers
            ),
            "logits_XY": error_metrics(logits[1][0], logits[2][0]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
