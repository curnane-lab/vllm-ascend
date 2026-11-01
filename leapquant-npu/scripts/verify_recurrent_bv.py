import argparse
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

from gdn_recurrent_candidates import fused_recurrent_gated_delta_rule_packed_decode_bv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--candidates", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20261031)
    B, H, HV, V, K = args.batch, 16, 32, 128, 128
    reference_state = torch.randn(B, HV, V, K, device="npu") * 0.02
    ssm_state_indices = torch.arange(1, B + 1, dtype=torch.int32, device="npu")
    A_log = torch.log(torch.rand(HV, device="npu") * 0.09 + 0.01)
    dt_bias = torch.randn(HV, device="npu") * 0.5
    steps = []
    for _ in range(args.windows * 16):
        mixed_qkv = torch.randn(B, 2 * H * K + HV * V, device="npu", dtype=torch.bfloat16)
        a = torch.randn(B, HV, device="npu", dtype=torch.bfloat16) * 0.1
        b = torch.randn(B, HV, device="npu", dtype=torch.bfloat16) * 0.1
        steps.append((mixed_qkv, a, b))

    payload = {"batch": B, "windows": args.windows, "steps_per_window": 16, "results": []}
    for candidate in args.candidates:
        state = reference_state.clone()
        windows = []
        status = "pass"
        for window in range(args.windows):
            max_output_abs = 0.0
            max_state_abs = 0.0
            max_output_l2 = 0.0
            max_state_l2 = 0.0
            for mixed_qkv, a, b in steps[window * 16 : (window + 1) * 16]:
                reference_out = torch.empty(B, 1, HV, V, device="npu", dtype=torch.bfloat16)
                candidate_out = torch.empty_like(reference_out)
                reference_out, reference_state = fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv, a, b, A_log, dt_bias, K**-0.5, reference_state,
                    reference_out, ssm_state_indices, use_qk_l2norm_in_kernel=True
                )
                candidate_out, state = fused_recurrent_gated_delta_rule_packed_decode_bv(
                    mixed_qkv, a, b, A_log, dt_bias, K**-0.5, state,
                    candidate_out, ssm_state_indices, use_qk_l2norm_in_kernel=True,
                    bv=candidate,
                )
                output_error = (candidate_out.float() - reference_out.float()).abs()
                state_error = (state - reference_state).abs()
                max_output_abs = max(max_output_abs, output_error.max().item())
                max_state_abs = max(max_state_abs, state_error.max().item())
                max_output_l2 = max(
                    max_output_l2,
                    (output_error.norm() / reference_out.float().norm().clamp_min(1e-30)).item(),
                )
                max_state_l2 = max(
                    max_state_l2,
                    (state_error.norm() / reference_state.norm().clamp_min(1e-30)).item(),
                )
                reference_ok = bool(torch.isfinite(reference_out).all() and torch.isfinite(reference_state).all())
                candidate_ok = bool(torch.isfinite(candidate_out).all() and torch.isfinite(state).all())
                if not reference_ok or not candidate_ok:
                    status = f"non_finite_reference_{not reference_ok}_candidate_{not candidate_ok}"
                    break
            windows.append(
                {
                    "window": window,
                    "max_output_abs": max_output_abs,
                    "max_state_abs": max_state_abs,
                    "max_output_relative_l2": max_output_l2,
                    "max_state_relative_l2": max_state_l2,
                }
            )
            if status != "pass":
                break
        payload["results"].append({"bv": candidate, "status": status, "windows": windows})

    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for result in payload["results"]:
        final = result["windows"][-1]
        print(
            f"BV={result['bv']} status={result['status']} output_abs={final['max_output_abs']:.6g} "
            f"state_abs={final['max_state_abs']:.6g} output_l2={final['max_output_relative_l2']:.6g} "
            f"state_l2={final['max_state_relative_l2']:.6g}"
        )
        if result["status"] == "pass":
            assert final["max_output_abs"] == 0.0
            assert final["max_state_abs"] == 0.0


if __name__ == "__main__":
    main()
