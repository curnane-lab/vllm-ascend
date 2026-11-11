from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(path: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def source_identity(path: str) -> dict:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "path": path,
        "sha": git(path, "rev-parse", "HEAD"),
        "branch": git(path, "status", "--short", "--branch", "--untracked-files=no"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_empty": not diff,
        "untracked_status": git(path, "status", "--porcelain"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/data/models/Qwen3.5-4B"))
    args = parser.parse_args()
    import fla_npu
    import torch
    import vllm

    import vllm_ascend

    distributions = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    fla_root = Path(fla_npu.__file__).parent
    opapi = fla_root / "opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    key_files = {
        "ascend_gdn": Path("/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py"),
        "ascend_device_op": Path("/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py"),
        "vllm_qwen_gdn": Path("/vllm-workspace/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"),
        "fla_stable": fla_root / "libfla_npu_stable.so",
        "fla_opapi": opapi,
    }
    npu_smi = subprocess.run(["npu-smi", "info"], check=True, text=True, stdout=subprocess.PIPE).stdout
    env_whitelist = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("ASCEND_", "VLLM_", "SPARSEGDN_", "REWY", "ReWY"))
        or key in ("PYTHONPATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    weight_files = [
        "model.safetensors-00001-of-00002.safetensors",
        "model.safetensors-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "config.json",
    ]
    model_hashes = {name: sha256(args.model / name) for name in weight_files}
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "E0+E1 only; no NPU kernel performance or E2 quality result",
        "hardware": {
            "npu_smi": npu_smi,
            "main_910c_status": "NOT-RUN; only 910B3 is present",
            "used_development_cards": [2, 3, 4, 5, 6],
            "unknown_jobs_terminated": False,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "distributions": distributions,
        "modules": {
            "vllm": vllm.__file__,
            "vllm_ascend": vllm_ascend.__file__,
            "torch": torch.__file__,
            "fla_npu": fla_npu.__file__,
        },
        "sources": {
            "production_vllm": source_identity("/vllm-workspace/vllm"),
            "production_ascend": source_identity("/vllm-workspace/vllm-ascend"),
            "experiment_worktree": source_identity("/data/vllm-ascend-sparsegdn-e0e1"),
        },
        "key_file_sha256": {name: sha256(path) for name, path in key_files.items()},
        "cann": {
            "toolkit_version_path": "/usr/local/Ascend/ascend-toolkit/latest/version.info",
            "toolkit_version": Path("/usr/local/Ascend/ascend-toolkit/latest/version.info").read_text()
            if Path("/usr/local/Ascend/ascend-toolkit/latest/version.info").exists()
            else subprocess.run(
                ["/usr/local/Ascend/cann-9.2.1/query_pkg_version.sh"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout,
            "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        },
        "model": {
            "path": str(args.model),
            "resolved_commit": "ed182e32090db791077e12e0f58d22f3daafa173",
            "dtype": "bfloat16",
            "recurrent_state_dtype": "float32",
            "sha256": model_hashes,
            "tokenizer": "actual local Qwen3.5-4B tokenizer",
        },
        "runtime": {
            "tp": 1,
            "chunked_prefill": True,
            "native_chunk_size": 64,
            "scheduler_max_tokens_per_step": 1024,
            "max_num_seqs": 64,
            "graph_mode": "FULL_DECODE_ONLY",
            "decode_full_capture_count": 7,
            "prefill_graph_mode": "not FULL replay; eager/chunked prefill trace",
            "prefix_caching_for_formal_baseline": False,
            "language_model_only": True,
            "rewy_or_unrelated_monkey_patch_detected": any(key.lower().startswith("rewy") for key in env_whitelist),
            "whitelisted_environment": env_whitelist,
        },
        "experiment": {
            "default_enabled": False,
            "commits_or_pushes": False,
            "native_sparse_kernel_started": False,
            "notes": "production sources read-only; experiment scripts are isolated and opt-in",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
