import argparse
import hashlib
import itertools
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.ops import gdn_pool_io as pool_io
from vllm_ascend.ops.gdn import _get_fla_gdn_prefill_op
from vllm_ascend.ops.gdn_attn_builder import prepare_chunk_indices
from vllm_ascend.ops.gdn_pool_prefill_ascendc import _binding, _opapi_candidates
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_

HK = 16
HV = 32
HEAD_DIM = 128
ROW_SIZE = HV * HEAD_DIM * HEAD_DIM
COUNTERS = {"pool": 0, "ascendc": 0, "gather": 0, "scatter": 0}
REAL_POOL_OP = pool_io._get_pool_prefill_op()
REAL_ASCENDC = pool_io.gdn_pool_prefill_ascendc
REAL_GATHER = pool_io.gather_ssm_states
REAL_SCATTER = pool_io.scatter_ssm_states_


def counted_pool(*args, **kwargs):
    COUNTERS["pool"] += 1
    return REAL_POOL_OP(*args, **kwargs)


def counted_ascendc(**kwargs):
    COUNTERS["ascendc"] += 1
    return REAL_ASCENDC(**kwargs)


def no_ascendc(**kwargs):
    COUNTERS["ascendc"] += 1
    return None


def counted_gather(*args, **kwargs):
    COUNTERS["gather"] += 1
    return REAL_GATHER(*args, **kwargs)


def counted_scatter(*args, **kwargs):
    COUNTERS["scatter"] += 1
    return REAL_SCATTER(*args, **kwargs)


def close(actual, expected, tolerance):
    try:
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        return True
    except AssertionError:
        return False


def metrics(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "finite": bool(torch.isfinite(actual.float()).all().item()),
        "bitwise": torch.equal(actual, expected),
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "close_1e_3": close(actual, expected, 1e-3),
        "close_1e_2": close(actual, expected, 1e-2),
    }


def make_state(initial, stride):
    if stride == ROW_SIZE:
        return initial.clone(), None
    storage = torch.full((*initial.shape[:2], stride), -13.0, dtype=initial.dtype, device=initial.device)
    state = storage[..., :ROW_SIZE].view(initial.shape)
    state.copy_(initial)
    return state, storage


def build_case(spec, pool_rows, seed):
    device = torch.device("npu:0")
    lengths = spec["lengths"]
    total = sum(lengths)
    batch = len(lengths)
    torch.manual_seed(seed)
    q = torch.randn(1, HK, total, HEAD_DIM, dtype=torch.bfloat16, device=device).transpose(1, 2)
    k = torch.randn(1, HK, total, HEAD_DIM, dtype=torch.bfloat16, device=device).transpose(1, 2)
    v = (0.1 * torch.randn(1, HV, total, HEAD_DIM, dtype=torch.bfloat16, device=device)).transpose(1, 2)
    g = -torch.rand(1, total, HV, dtype=torch.float32, device=device)
    beta = torch.rand(1, total, HV, dtype=torch.bfloat16, device=device)
    cu_host = (0, *itertools.accumulate(lengths))
    cu_cpu = torch.tensor(cu_host, dtype=torch.int32)
    src = torch.arange(3, 3 + batch, dtype=torch.int32, device=device)
    dst = src.clone() if spec["alias"] else torch.arange(17, 17 + batch, dtype=torch.int32, device=device)
    initial = 0.05 * torch.randn(1, pool_rows, HV, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=device)
    reference_state, reference_storage = make_state(initial, spec["stride"])
    actual_state, actual_storage = make_state(initial, spec["stride"])
    reference_state = reference_state[0]
    actual_state = actual_state[0]
    data = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "cu": torch.tensor(cu_host, dtype=torch.int32, device=device),
        "src": src,
        "dst": dst,
        "has_initial": torch.tensor(spec["has_initial"], dtype=torch.bool, device=device),
        "scale": HEAD_DIM**-0.5,
        "meta": SimpleNamespace(
            cu_seqlens_host=cu_host,
            chunk_indices_chunk64_host=tuple(prepare_chunk_indices(cu_cpu, 64).reshape(-1).tolist()),
            cu_seqlens_kern=None,
            keep_meta=None,
        ),
    }
    return data, initial, reference_state, reference_storage, actual_state, actual_storage


def production_reference(data, state):
    initial = gather_ssm_states(state, data["src"], data["has_initial"])
    output, final = DeviceOperator.fla_gdn_prefill(
        q=data["q"],
        k=data["k"],
        v=data["v"],
        g=data["g"],
        beta=data["beta"],
        initial_state=initial,
        scale=data["scale"],
        prebuilt_meta=data["meta"],
        fused_fwd=chunk_gated_delta_rule_fwd,
    )
    scatter_ssm_states_(state, data["dst"], final.to(state.dtype).contiguous())
    return output, initial


def dispatch_expectations(mode):
    if mode == "pool":
        return {"pool": 1, "ascendc": 0, "gather": 0, "scatter": 0}
    return {"pool": 0, "ascendc": 1, "gather": 1, "scatter": 1}


def configure_mode(mode):
    if mode == "pool":
        pool_io._get_pool_prefill_op = lambda: counted_pool
        pool_io.gdn_pool_prefill_ascendc = counted_ascendc
    else:
        pool_io._get_pool_prefill_op = lambda: None
        pool_io.gdn_pool_prefill_ascendc = no_ascendc


def run_case(spec, mode, pool_rows, seed):
    data, initial, reference_state, reference_storage, actual_state, actual_storage = build_case(spec, pool_rows, seed)
    expected, reference_initial = production_reference(data, reference_state)
    for key in COUNTERS:
        COUNTERS[key] = 0
    configure_mode(mode)
    actual = pool_io.run_indexed_pool_prefill(
        data["q"],
        data["k"],
        data["v"],
        data["g"],
        data["beta"],
        actual_state,
        data["src"],
        data["has_initial"],
        data["scale"],
        dst_slots=data["dst"],
        cu_seqlens=data["cu"],
    )
    torch.npu.synchronize()
    if actual is None:
        raise RuntimeError(f"{mode} unexpectedly rejected {spec}")
    output_metrics = metrics(actual, expected)
    state_metrics = metrics(actual_state, reference_state)
    calls = dict(COUNTERS)
    expected_calls = dispatch_expectations(mode)
    source_preserved = spec["alias"] or torch.equal(actual_state[data["src"].long()], initial[0, data["src"].long()])
    padding_preserved = actual_storage is None or bool((actual_storage[..., ROW_SIZE:] == -13.0).all().item())
    reference_padding_preserved = reference_storage is None or bool(
        (reference_storage[..., ROW_SIZE:] == -13.0).all().item()
    )
    fresh = not all(spec["has_initial"])
    fresh_reference_zero = not fresh or bool((reference_initial[~data["has_initial"]] == 0).all().item())
    passed = (
        output_metrics["close_1e_2"]
        and state_metrics["close_1e_2"]
        and output_metrics["finite"]
        and state_metrics["finite"]
        and calls == expected_calls
        and source_preserved
        and padding_preserved
        and reference_padding_preserved
        and fresh_reference_zero
    )
    passed_strict = passed and output_metrics["close_1e_3"] and state_metrics["close_1e_3"]
    return {
        "name": spec["name"],
        "mode": mode,
        "lengths": spec["lengths"],
        "has_initial": spec["has_initial"],
        "src_equals_dst": spec["alias"],
        "pool_stride_elements": spec["stride"],
        "q_shape": list(data["q"].shape),
        "q_stride": list(data["q"].stride()),
        "output": output_metrics,
        "state": state_metrics,
        "calls": calls,
        "expected_calls": expected_calls,
        "source_preserved": source_preserved,
        "padding_preserved": padding_preserved and reference_padding_preserved,
        "fresh_reference_zero": fresh_reference_zero,
        "pass_1e_2": passed,
        "pass_1e_3": passed_strict,
    }


def failure_injection(pool_rows, seed):
    spec = {
        "name": "cached_alias_failure",
        "lengths": [65],
        "has_initial": [True],
        "alias": True,
        "stride": PADDED_STRIDE,
    }
    data, _, reference_state, _, actual_state, _ = build_case(spec, pool_rows, seed)
    expected, _ = production_reference(data, reference_state)
    pool_io._get_pool_prefill_op = lambda: None
    for key in COUNTERS:
        COUNTERS[key] = 0

    def poison_after_launch(**kwargs):
        COUNTERS["ascendc"] += 1
        dst_pool = kwargs["dst_pool"]
        dst_slots = kwargs["dst_slots"]
        dst_pool.index_add_(0, dst_slots.long(), torch.ones_like(dst_pool[dst_slots.long()]))
        raise RuntimeError("aclnnGdnPoolPrefill failed; the destination pool may be partially written")

    pool_io.gdn_pool_prefill_ascendc = poison_after_launch
    returned = True
    error = None
    try:
        actual = pool_io.run_indexed_pool_prefill(
            data["q"],
            data["k"],
            data["v"],
            data["g"],
            data["beta"],
            actual_state,
            data["src"],
            data["has_initial"],
            data["scale"],
            dst_slots=data["dst"],
            cu_seqlens=data["cu"],
        )
        torch.npu.synchronize()
    except Exception as exception:
        returned = False
        error = repr(exception)
    result = {
        "scenario": "Simulate an AscendC partial write followed by a launch exception, with src==dst.",
        "fallback_returned": returned,
        "exception": error,
        "calls": dict(COUNTERS),
        "fallback_not_attempted": COUNTERS["gather"] == 0 and COUNTERS["scatter"] == 0,
        "partial_write_error_propagated": not returned and error is not None and "partially written" in error,
    }
    if returned:
        result["output"] = metrics(actual, expected)
        result["state"] = metrics(actual_state, reference_state)
        result["silent_wrong_result"] = not (result["output"]["close_1e_2"] and result["state"]["close_1e_2"])
    return result


def cann_version():
    path = Path("/usr/local/Ascend/cann-9.2.0/aarch64-linux/ascend_toolkit_install.info")
    if not path.is_file():
        return None
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def main():
    """Compare the available pool adapters with the production FLA path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--pool-rows", type=int, default=32)
    parser.add_argument("--pool-stride", type=int, default=786432)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--baseline", default="f1b41df375ec86d513f22782a146fb63a2425d42")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    global PADDED_STRIDE
    PADDED_STRIDE = args.pool_stride
    torch.npu.set_device(args.device)
    if REAL_POOL_OP is None:
        raise RuntimeError("chunk_gated_delta_rule_pool is unavailable")
    if _get_fla_gdn_prefill_op() is not chunk_gated_delta_rule_fwd:
        raise RuntimeError("reference fused_fwd differs from the production import")
    pool_io.gather_ssm_states = counted_gather
    pool_io.scatter_ssm_states_ = counted_scatter
    cases = []
    for length in (65, 130):
        for fresh in (False, True):
            for alias in (False, True):
                initial = not fresh
                cases.append(
                    {
                        "name": f"T{length}_{'fresh' if fresh else 'cached'}_{'alias' if alias else 'independent'}",
                        "lengths": [length],
                        "has_initial": [initial],
                        "alias": alias,
                        "stride": args.pool_stride,
                        "modes": ("pool", "fallback"),
                    }
                )
    cases.extend(
        [
            {
                "name": "packed_T65_T7_mixed_independent",
                "lengths": [65, 7],
                "has_initial": [True, False],
                "alias": False,
                "stride": args.pool_stride,
                "modes": ("pool",),
            },
            {
                "name": "packed_T128x3_cached_independent",
                "lengths": [128, 128, 128],
                "has_initial": [True, True, True],
                "alias": False,
                "stride": args.pool_stride,
                "modes": ("pool",),
            },
            {
                "name": "T7_fresh_independent",
                "lengths": [7],
                "has_initial": [False],
                "alias": False,
                "stride": args.pool_stride,
                "modes": ("pool", "fallback"),
            },
            {
                "name": "T65_cached_contiguous",
                "lengths": [65],
                "has_initial": [True],
                "alias": False,
                "stride": ROW_SIZE,
                "modes": ("pool", "fallback"),
            },
        ]
    )
    results = []
    for index, spec in enumerate(cases):
        for mode in spec.pop("modes"):
            try:
                result = run_case(spec, mode, args.pool_rows, args.seed + index)
            except Exception as exception:
                result = {
                    "name": spec["name"],
                    "mode": mode,
                    "error": repr(exception),
                    "traceback": traceback.format_exc(),
                    "pass_1e_2": False,
                    "pass_1e_3": False,
                }
            results.append(result)
            print(json.dumps(result), flush=True)
            torch.npu.empty_cache()
    injection = failure_injection(args.pool_rows, args.seed + 100)
    fp32_state = torch.zeros(4, HV, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="npu:0")
    fp32_indices = torch.zeros(1, dtype=torch.int32, device="npu:0")
    fp32_initial = torch.zeros(1, dtype=torch.bool, device="npu:0")
    dtype_result = {
        "fp32_supports_pool_io": pool_io.supports_pool_io(fp32_state, fp32_indices, fp32_initial, fp32_state.device),
        "bf16_supports_pool_io": pool_io.supports_pool_io(
            fp32_state.bfloat16(), fp32_indices, fp32_initial, fp32_state.device
        ),
        "scope": "Dispatch gate check here; real-server FP32 behavior requires a separate service arm.",
    }
    passed = sum(result["pass_1e_2"] for result in results)
    strict_passed = sum(result["pass_1e_3"] for result in results)
    library = Path(pool_io.__file__).parent
    import sgl_kernel_npu

    operator_library = Path(sgl_kernel_npu.__file__).parent / "lib/libsgl_kernel_npu.so"
    result = {
        "baseline": args.baseline,
        "environment": {
            "device": torch.npu.get_device_name(0),
            "hardware_profile": str(get_current_hardware_profile()),
            "reference": DeviceOperator.fla_gdn_prefill.__qualname__,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": cann_version(),
            "sgl_kernel_npu": sgl_kernel_npu.__file__,
            "operator_library_sha256": hashlib.sha256(operator_library.read_bytes()).hexdigest(),
            "pool_io": str(library),
        },
        "ascendc": {"candidates": _opapi_candidates(), "binding_available": _binding() is not None},
        "cases": results,
        "failure_injection": injection,
        "dtype": dtype_result,
        "summary": {
            "total_cases": len(results),
            "passed_1e_2": passed,
            "passed_1e_3": strict_passed,
            "a5_executed": False,
            "a5_note": "Only Ascend910B3 hardware is available in this environment.",
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(injection, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
