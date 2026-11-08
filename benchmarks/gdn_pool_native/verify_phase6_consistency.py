import argparse
import hashlib
import json
from pathlib import Path

import fla_npu
import torch
import torch_npu
import verify_pool_production_contract as base
from fla_npu.ops.ascendc import _stable, npu_chunk_gated_delta_rule_fwd_pool
from verify_phase6_pool import case_specs
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd


def error(actual, expected):
    return {
        "bitwise": torch.equal(actual, expected),
        "max_abs": (actual.float() - expected.float()).abs().max().item(),
        "finite": bool(torch.isfinite(actual.float()).all().item()),
    }


def pool_call(data, state, backend):
    query = l2norm_fwd(data["q"].contiguous()).transpose(1, 2).contiguous()
    key = l2norm_fwd(data["k"].contiguous()).transpose(1, 2).contiguous()
    value = data["v"].transpose(1, 2).contiguous()
    if backend == "ctypes":
        import ctypes

        from fla_npu.ops.ascendc import _aclnn_ctypes

        output = _aclnn_ctypes._empty((query.shape[0], query.shape[2], value.shape[1], value.shape[-1]), value)
        return _aclnn_ctypes._runtime_call_aclnn(
            "aclnnChunkGatedDeltaRuleFwdPool",
            lambda ctx: [
                ctx.tensor(query, "q"),
                ctx.tensor(key, "k"),
                ctx.tensor(value, "v"),
                ctx.tensor(data["g"].contiguous(), "g"),
                ctx.tensor(data["beta"].contiguous(), "beta"),
                ctx.tensor(state, "pool"),
                ctx.int_array(data["meta"].cu_seqlens_host),
                ctx.int_array(data["meta"].chunk_indices_chunk64_host),
                ctx.tensor(data["src"], "source"),
                ctx.tensor(data["dst"], "destination"),
                ctx.tensor(data["has_initial"], "has_initial"),
                ctypes.c_double(data["scale"]),
                ctypes.c_int64(64),
                ctx.tensor(output, "output"),
                ctx.tensor(state, "pool_output"),
            ],
            output,
        )
    return npu_chunk_gated_delta_rule_fwd_pool(
        query,
        key,
        value,
        data["g"].contiguous(),
        data["beta"].contiguous(),
        state,
        data["src"],
        data["dst"],
        data["has_initial"],
        cu_seqlens=data["meta"].cu_seqlens_host,
        chunk_indices=data["meta"].chunk_indices_chunk64_host,
        scale=data["scale"],
        chunk_size=64,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--backend", choices=("stable", "ctypes"), default="stable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device("npu:0")
    records = []
    for index, spec in enumerate(case_specs()):
        if args.only and spec["name"] != args.only:
            continue
        seed = 20261109 + index
        data, initial, reference_state, _, actual_state, actual_storage = base.build_case(spec, 96, seed)
        data["dst"] = torch.arange(40, 40 + len(spec["lengths"]), dtype=torch.int32, device="npu")
        if spec["alias"]:
            data["dst"] = data["src"].clone()
        reference_outputs = []
        reference_states = []
        for _ in range(2):
            state = reference_state.clone()
            output, _ = base.production_reference(data, state)
            torch.npu.synchronize()
            reference_outputs.append(output)
            reference_states.append(state)
        pool_snapshot = actual_state.clone()
        try:
            actual = pool_call(data, actual_state, args.backend)
            torch.npu.synchronize()
        except Exception as exception:
            record = {
                "case": spec["name"],
                "seed": seed,
                "stage": "pool_call",
                "error": repr(exception),
                "passed": False,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            torch.npu.empty_cache()
            continue
        mask = torch.ones(actual_state.shape[0], dtype=torch.bool, device="npu")
        mask[data["src"].long()] = False
        mask[data["dst"].long()] = False
        record = {
            "case": spec["name"],
            "seed": seed,
            "baseline_output_repeat": error(reference_outputs[0], reference_outputs[1]),
            "baseline_state_repeat": error(reference_states[0], reference_states[1]),
            "pool_output_vs_baseline": error(actual, reference_outputs[0]),
            "pool_state_vs_baseline": error(actual_state, reference_states[0]),
            "pool_state_vs_transposed_baseline": error(
                actual_state[data["dst"].long()], reference_states[0][data["dst"].long()].transpose(-1, -2)
            ),
            "source_preserved": spec["alias"]
            or torch.equal(actual_state[data["src"].long()], pool_snapshot[data["src"].long()]),
            "unselected_preserved": torch.equal(actual_state[mask], pool_snapshot[mask]),
            "changed_slots": (actual_state != pool_snapshot).any(dim=(1, 2, 3)).nonzero().flatten().cpu().tolist(),
            "padding_preserved": actual_storage is None
            or bool((actual_storage[..., base.ROW_SIZE :] == -13.0).all().item()),
        }
        record["passed"] = (
            record["baseline_output_repeat"]["bitwise"]
            and record["baseline_state_repeat"]["bitwise"]
            and record["pool_output_vs_baseline"]["bitwise"]
            and record["pool_state_vs_baseline"]["bitwise"]
            and record["source_preserved"]
            and record["unselected_preserved"]
            and record["padding_preserved"]
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        torch.npu.empty_cache()
    result = {
        "comparison": (
            "Direct production chunk_gated_delta_rule_fwd plus gather/scatter versus the "
            "Phase6 indexed pool entry, identical normalized inputs."
        ),
        "requirements": (
            "bitwise output and full-pool state equality; source, unselected slots and pool padding preserved"
        ),
        "library": _stable._lib_path(),
        "library_sha256": hashlib.sha256(Path(_stable._lib_path()).read_bytes()).hexdigest(),
        "fla_npu": fla_npu.__file__,
        "torch_npu": torch_npu.__version__,
        "backend": args.backend,
        "passed": sum(record["passed"] for record in records),
        "total": len(records),
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "total": result["total"]}))
    print(args.output)
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
