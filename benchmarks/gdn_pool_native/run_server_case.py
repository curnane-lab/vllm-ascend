import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_DIR / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

benchmark = importlib.import_module("bench_gdn_pool_prefill_server")

RUNTIME = Path(os.environ["GDN_PHASE6_RUNTIME"])
OP_API = RUNTIME / "fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
POOL_OP_API = OP_API.with_name("libcust_pool_opapi.so")
OPP = RUNTIME / "fla_npu/opp/vendors/fla_npu_transformer"
FULL_OP_API = Path(
    os.environ.get(
        "GDN_PHASE6_FULL_OP_API",
        str(RUNTIME / "full_vendor/op_api/lib/libcust_opapi.so"),
    )
)
FULL_OPP = FULL_OP_API.parent.parent
VLLM_OPP = os.environ.get("GDN_VLLM_OPP")
VLLM_OP_API = Path(VLLM_OPP) / "op_api/lib/libcust_opapi.so" if VLLM_OPP else None
STABLE = RUNTIME / "libfla_npu_stable.so"


def file_sha256(path):
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    original_popen = benchmark.subprocess.Popen

    def popen(command, *args, **kwargs):
        environment = kwargs.get("env")
        if environment is not None:
            custom_paths = [OPP, FULL_OPP]
            if VLLM_OPP:
                custom_paths.append(Path(VLLM_OPP))
            if environment.get("ASCEND_CUSTOM_OPP_PATH"):
                custom_paths.append(environment["ASCEND_CUSTOM_OPP_PATH"])
            environment.update(
                {
                    "FLA_NPU_STABLE_STREAM": "accessor",
                    "FLA_NPU_STABLE_LIB": str(STABLE),
                    "FLA_NPU_OP_API_LIB": str(OP_API),
                    "FLA_NPU_STABLE_TRACE": "1",
                    "GDN_PHASE6_FULL_OPP": "1",
                    "GDN_PHASE6_FULL_OP_API": str(FULL_OP_API),
                    "ASCEND_CUSTOM_OPP_PATH": ":".join(map(str, custom_paths)),
                    "PYTHONPATH": (
                        f"{os.environ.get('GDN_PHASE6_SITE_DIR', str(Path(__file__).parent / 'phase6_runtime'))}:"
                        f"{environment.get('PYTHONPATH', '')}"
                    ),
                    "LD_LIBRARY_PATH": (
                        f"{REPO_DIR / 'vllm_ascend'}:{REPO_DIR / 'vllm_ascend/lib'}:"
                        f"{OP_API.parent}:{environment.get('LD_LIBRARY_PATH', '')}"
                    ),
                }
            )
            for name in ("GDN_INSTRUMENT_FILE", "GDN_PROFILE_FILE", "GDN_POOL_DIAG_FILE", "GDN_POOL_PROTOTYPE"):
                environment.pop(name, None)
            environment.pop("GDN_GRAPH_REPLAY_FILE", None)
            if any("FULL_AND_PIECEWISE" in argument for argument in command):
                environment["GDN_GRAPH_REPLAY_FILE"] = str(output.with_suffix(".replay.log"))
            if os.environ.get("GDN_SYNC_SCHEDULING") == "1":
                command.append("--no-async-scheduling")
            if os.environ.get("GDN_DISABLE_PREFIX_CACHE") == "1":
                command = [argument for argument in command if argument != "--enable-prefix-caching"]
                command.append("--disable-prefix-caching")
        return original_popen(command, *args, **kwargs)

    benchmark.subprocess.Popen = popen
    benchmark.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    server_log = output.with_suffix(".server.log").read_text(encoding="utf-8")
    native_trace = "aclnnChunkGatedDeltaRuleFwdPool workspace=" in server_log
    if result["mode"] == "pool" and not native_trace:
        raise RuntimeError("pool mode did not invoke the native pool operator")
    binary_config = json.loads(
        (OPP / "op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json").read_text()
    )
    binary = (
        OPP / "op_impl/ai_core/tbe/kernel" / binary_config["ChunkGatedDeltaRuleFwdPool"]["binaryList"][0]["binPath"]
    )
    result["phase6_pool"] = {
        "operator": "fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_fwd_pool",
        "stable_stream": "accessor",
        "sync_scheduling": os.environ.get("GDN_SYNC_SCHEDULING") == "1",
        "prefix_cache": os.environ.get("GDN_DISABLE_PREFIX_CACHE") != "1",
        "fla_backend": os.environ.get("FLA_NPU_STABLE_ABI", "stable"),
        "stable_library": str(STABLE),
        "stable_library_sha256": file_sha256(STABLE),
        "op_api": str(OP_API),
        "op_api_sha256": file_sha256(OP_API),
        "pool_op_api": str(POOL_OP_API),
        "pool_op_api_sha256": file_sha256(POOL_OP_API),
        "kernel": str(binary),
        "kernel_sha256": file_sha256(binary),
        "native_operator_observed": native_trace,
        "full_op_api_preload": str(FULL_OP_API),
        "full_op_api_sha256": file_sha256(FULL_OP_API),
        "vllm_op_api": str(VLLM_OP_API) if VLLM_OP_API else None,
        "vllm_op_api_sha256": file_sha256(VLLM_OP_API),
        "layout_fusion": json.loads(os.environ.get("VLLM_ASCEND_GDN_POOL_LAYOUT_FUSION", "0")) == 1,
        "cow_elision": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
