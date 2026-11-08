import argparse
import itertools
import json
from pathlib import Path

import torch
import verify_pool_production_contract as base

from vllm_ascend.ops import gdn_pool_io as pool_io
from vllm_ascend.ops.gdn_attn_builder import prepare_chunk_indices


def case_specs():
    specs = [
        ("B1_T64_cached_contiguous", [64], [True], False, base.ROW_SIZE),
        ("B1_T65_cached_padded", [65], [True], False, 786432),
        ("B1_T128_cached_padded", [128], [True], False, 786432),
        ("B1_T65_cached_alias", [65], [True], True, 786432),
        ("B1_T65_fresh", [65], [False], False, 786432),
        ("B1_T7_fresh", [7], [False], False, 786432),
        ("B2_T65_cached", [65, 65], [True, True], False, base.ROW_SIZE),
        ("B2_T128_cached_padded", [128, 128], [True, True], False, 786432),
        ("B2_T65_fresh", [65, 65], [False, False], False, 786432),
        ("B2_T65_mixed", [65, 65], [True, False], False, 786432),
        ("B2_T64_mixed_alias", [64, 64], [True, False], True, 786432),
        ("B7_T65_cached", [65] * 7, [True] * 7, False, 786432),
        ("B7_T128_mixed", [128] * 7, [True, False, True, True, False, True, False], False, 786432),
        ("B32_T65_cached", [65] * 32, [True] * 32, False, 786432),
        ("B32_T128_cached", [128] * 32, [True] * 32, False, 786432),
        ("B1_T64_fresh_contiguous", [64], [False], False, base.ROW_SIZE),
        ("B1_T64_fresh_padded", [64], [False], False, 786432),
    ]
    return [
        {"name": name, "lengths": lengths, "has_initial": initial, "alias": alias, "stride": stride}
        for name, lengths, initial, alias, stride in specs
    ]


def phase6_pool(query, key, value, pool, src, dst, *, beta, actual_seq_lengths, scale, g, has_initial_state):
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd_pool

    base.COUNTERS["pool"] += 1
    lengths = actual_seq_lengths.cpu().tolist()
    cu = (0, *itertools.accumulate(lengths))
    chunk = tuple(prepare_chunk_indices(torch.tensor(cu, dtype=torch.int32), 64).reshape(-1).tolist())
    q = query.unsqueeze(0).transpose(1, 2).contiguous()
    k = key.unsqueeze(0).transpose(1, 2).contiguous()
    v = value.unsqueeze(0).transpose(1, 2).contiguous()
    output = npu_chunk_gated_delta_rule_fwd_pool(
        q,
        k,
        v,
        g.unsqueeze(0).contiguous(),
        beta.unsqueeze(0).contiguous(),
        pool,
        src,
        dst,
        has_initial_state,
        cu_seqlens=cu,
        chunk_indices=chunk,
        scale=scale,
        chunk_size=64,
    )
    return output.squeeze(0)


def error(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "bitwise": torch.equal(actual, expected),
        "max_abs": difference.max().item(),
        "finite": bool(torch.isfinite(actual.float()).all().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "check"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--only")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    saved = torch.load(args.baseline, map_location="cpu") if args.mode == "check" else {}
    records = []
    for index, spec in enumerate(case_specs()):
        if args.only and spec["name"] != args.only:
            continue
        seed = 20261109 + index
        data, initial, reference_state, _, actual_state, actual_storage = base.build_case(spec, 96, seed)
        if not spec["alias"]:
            data["dst"] = torch.arange(40, 40 + len(spec["lengths"]), dtype=torch.int32, device=data["dst"].device)
        record = {"case": spec["name"], "seed": seed}
        if args.mode == "baseline":
            expected, _ = base.production_reference(data, reference_state)
            torch.npu.synchronize()
            saved[spec["name"]] = {
                "output": expected.cpu(),
                "final": reference_state[data["dst"].long()].cpu(),
            }
        else:
            expected, _ = base.production_reference(data, reference_state)
            torch.npu.synchronize()
            old = saved[spec["name"]]
            record["default_output_vs_original"] = error(expected.cpu(), old["output"])
            record["default_final_vs_original"] = error(reference_state[data["dst"].long()].cpu(), old["final"])
            for key in base.COUNTERS:
                base.COUNTERS[key] = 0
            pool_io._get_pool_prefill_op = lambda: phase6_pool
            pool_io.gdn_pool_prefill_ascendc = base.no_ascendc
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
            record["pool_output_vs_default"] = error(actual.cpu(), expected.cpu())
            record["pool_final_vs_default"] = error(actual_state.cpu(), reference_state.cpu())
            record["pool_output_vs_original"] = error(actual.cpu(), old["output"])
            record["pool_final_vs_original"] = error(actual_state[data["dst"].long()].cpu(), old["final"])
            record["calls"] = dict(base.COUNTERS)
            record["source_preserved"] = spec["alias"] or torch.equal(
                actual_state[data["src"].long()], initial[0, data["src"].long()]
            )
            mask = torch.ones(actual_state.shape[0], dtype=torch.bool, device=actual_state.device)
            mask[data["src"].long()] = False
            mask[data["dst"].long()] = False
            record["unselected_preserved"] = torch.equal(actual_state[mask], initial[0, mask])
            record["padding_preserved"] = actual_storage is None or bool(
                (actual_storage[..., base.ROW_SIZE :] == -13.0).all().item()
            )
            record["passed"] = (
                all(
                    record[key]["bitwise"]
                    for key in (
                        "default_output_vs_original",
                        "default_final_vs_original",
                        "pool_output_vs_default",
                        "pool_final_vs_default",
                        "pool_output_vs_original",
                        "pool_final_vs_original",
                    )
                )
                and record["source_preserved"]
                and record["unselected_preserved"]
                and record["padding_preserved"]
                and record["calls"] == {"pool": 1, "ascendc": 0, "gather": 0, "scatter": 0}
            )
        records.append(record)
        print(json.dumps(record), flush=True)
        del data, initial, reference_state, actual_state, actual_storage
        torch.npu.empty_cache()
    if args.mode == "baseline":
        torch.save(saved, args.baseline)
    args.output.write_text(json.dumps({"mode": args.mode, "cases": records}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
