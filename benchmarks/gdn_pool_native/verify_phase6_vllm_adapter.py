import json
from pathlib import Path

import torch
import verify_pool_production_contract as base
from verify_phase6_pool import case_specs

from vllm_ascend.ops import gdn_pool_io as pool_io


def main():
    torch.npu.set_device("npu:0")
    pool_io._get_pool_prefill_op = lambda: None
    records = []
    for index in (0, 1, 9):
        spec = case_specs()[index]
        data, _, reference_state, _, actual_state, _ = base.build_case(spec, 96, 20261109 + index)
        data["dst"] = torch.arange(40, 40 + len(spec["lengths"]), dtype=torch.int32, device="npu")
        if spec["alias"]:
            data["dst"] = data["src"].clone()
        expected, _ = base.production_reference(data, reference_state)
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
            chunk_metadata=data["meta"],
        )
        torch.npu.synchronize()
        record = {
            "case": spec["name"],
            "q_shape": list(data["q"].shape),
            "pool_shape": list(actual_state.shape),
            "pool_stride": list(actual_state.stride()),
            "output_bitwise": torch.equal(actual, expected),
            "state_bitwise": torch.equal(actual_state, reference_state),
        }
        record["passed"] = record["output_bitwise"] and record["state_bitwise"]
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {"passed": sum(record["passed"] for record in records), "total": len(records), "records": records}
    output = Path("results/phase6_vllm_adapter_consistency.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
