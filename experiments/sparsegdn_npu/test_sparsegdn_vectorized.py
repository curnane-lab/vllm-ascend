from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sparsegdn_reference import (
    IntraOutputs,
    _block_coefficients,
    _delta_matrix,
    _intra_chunk,
    _materialize_w,
    error_metrics,
    make_inputs,
    pad_inputs,
    run_reference,
)
from sparsegdn_vectorized import _chunked_inputs, run_vectorized_reference


def check_deferred_w() -> dict:
    inputs = make_inputs(64, 20265001, key_dim=8, num_key_heads=2, num_value_heads=4)
    padded = pad_inputs(inputs)
    k = inputs.k[:, 0]
    v = inputs.v[:, 0]
    g = padded.g[0, :, 0]
    beta = inputs.beta[:, 0]
    matrix = _delta_matrix(k, g, beta)
    diagonal, adjacent = _block_coefficients(matrix)
    u, read_coefficients = _intra_chunk(k, v, g, beta, "block-adjacent")
    chunked = _chunked_inputs(inputs)
    intra = IntraOutputs(
        torch.zeros(1, 64, 4, 8, dtype=u.dtype),
        read_coefficients.unsqueeze(0).unsqueeze(0),
        torch.zeros(1, 4, dtype=u.dtype),
        "block-adjacent",
    )
    actual = _materialize_w(chunked, intra, 0, 0)
    coefficients = torch.zeros(64, 64, dtype=u.dtype)
    for block in range(4):
        current = slice(block * 16, (block + 1) * 16)
        coefficients[current, current] = diagonal[block]
        if block:
            previous = slice((block - 1) * 16, block * 16)
            coefficients[current, previous] = adjacent[block - 1]
    expected = coefficients @ ((beta * torch.exp2(g))[:, None] * k)
    result = error_metrics(actual, expected)
    result["block_errors"] = [
        error_metrics(actual[index * 16 : (index + 1) * 16], expected[index * 16 : (index + 1) * 16])
        for index in range(4)
    ]
    result["passed"] = result["relative_frobenius"] <= 1e-8
    return result


def compare_driver(length: int, key_dim: int) -> dict:
    inputs = make_inputs(length, 20265002 + length + key_dim, key_dim=key_dim)
    slow = run_reference(inputs, return_state_trace=True)
    fast = run_vectorized_reference(inputs, return_state_trace=True)
    result = {
        "length": length,
        "key_dim": key_dim,
        "mask_differences": int((slow.selector.mask != fast.selector.mask).sum()),
        "tier_differences": int((slow.selector.tiers != fast.selector.tiers).sum()),
        "U": error_metrics(fast.intra.u, slow.intra.u),
        "corrected_U": error_metrics(fast.corrected_u, slow.corrected_u),
        "output": error_metrics(fast.output, slow.output),
        "final_state": error_metrics(fast.final_state, slow.final_state),
    }
    result["passed"] = (
        result["mask_differences"] == 0
        and result["tier_differences"] == 0
        and result["U"]["relative_frobenius"] <= 1e-8
        and result["corrected_U"]["relative_frobenius"] <= 1e-8
        and result["output"]["relative_frobenius"] <= 1e-8
        and result["final_state"]["relative_frobenius"] <= 1e-8
    )
    return result


def check_large_decay() -> dict:
    inputs = make_inputs(256, 20265999, key_dim=128, dtype=torch.float32)
    inputs.log_decay.fill_(-4.0)
    slow = run_reference(inputs, return_state_trace=True)
    fast = run_vectorized_reference(inputs, return_state_trace=True)
    result = {
        "scenario": "log2 decay=-4 per token; future exponent exceeds FP32 range",
        "slow_finite": bool(slow.output.isfinite().all() and slow.final_state.isfinite().all()),
        "fast_finite": bool(fast.output.isfinite().all() and fast.final_state.isfinite().all()),
        "mask_differences": int((slow.selector.mask != fast.selector.mask).sum()),
        "output": error_metrics(fast.output, slow.output),
        "final_state": error_metrics(fast.final_state, slow.final_state),
    }
    result["passed"] = (
        result["slow_finite"]
        and result["fast_finite"]
        and result["mask_differences"] == 0
        and result["output"]["relative_frobenius"] <= 1e-5
        and result["final_state"]["relative_frobenius"] <= 1e-5
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "deferred_w_all_blocks": check_deferred_w(),
        "driver_comparisons": [compare_driver(130, 8), compare_driver(320, 128)],
        "large_decay": check_large_decay(),
    }
    report["passed"] = (
        report["deferred_w_all_blocks"]["passed"]
        and report["large_decay"]["passed"]
        and all(case["passed"] for case in report["driver_comparisons"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
