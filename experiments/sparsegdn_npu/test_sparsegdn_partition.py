from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from sparsegdn_reference import (
    GDNInputs,
    _value_head_keys,
    adaptive_partition_policy,
    error_metrics,
    make_inputs,
    run_frozen_partition,
    run_reference,
    slice_inputs,
)


def make_shared_prefix_inputs(prefix: int, suffix: int, seed: int, key_dim: int):
    full_x = make_inputs(prefix + suffix, seed, key_dim=key_dim)
    full_y = make_inputs(prefix + suffix, seed + 1, key_dim=key_dim)
    for name in ("q", "k", "v", "log_decay", "beta", "z"):
        getattr(full_y, name)[:prefix] = getattr(full_x, name)[:prefix]
    full_y.norm_scale = full_x.norm_scale
    return full_x, full_y


def state_at_token(inputs: GDNInputs, result, token: int) -> torch.Tensor:
    if result.state_trace is None:
        raise ValueError("state trace is required")
    if token == inputs.length:
        return result.final_state
    chunk, offset = divmod(token, 64)
    if offset == 0:
        return result.state_trace[chunk]
    state = result.state_trace[chunk]
    start = chunk * 64
    local_decay = inputs.log_decay[start:token].cumsum(dim=0)
    gamma = local_decay[-1]
    k = _value_head_keys(inputs.k, inputs.num_value_heads)[start:token]
    corrected = result.corrected_u[chunk, :offset]
    selected = result.selector.mask[:, chunk].repeat_interleave(2)
    weights = torch.exp2(gamma[None] - local_decay)[:, :, None]
    writes = torch.einsum("lhk,lhv->hkv", k, corrected * weights)
    writes = writes * selected[:, None, None]
    return torch.exp2(gamma)[:, None, None] * state + writes


def piece_lengths(total: int, maximum: int) -> list[int]:
    pieces = []
    remaining = total
    while remaining:
        current = min(maximum, remaining)
        pieces.append(current)
        remaining -= current
    return pieces


def proxy_weight(inputs: GDNInputs) -> torch.Tensor:
    features = inputs.num_value_heads * inputs.v.shape[-1]
    generator = torch.Generator().manual_seed(20261111 + features)
    return torch.randn(
        features,
        16,
        generator=generator,
        dtype=inputs.v.dtype,
    ) / math.sqrt(features)


def proxy_logits(output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return output[-1].flatten() @ weight


def compare_valid_u(prefix_result, full_result, prefix: int) -> dict:
    chunks, remainder = divmod(prefix, 64)
    differences = []
    expected_norm = 0.0
    actual_norm = 0.0
    for chunk in range(chunks):
        actual = prefix_result.intra.u[chunk]
        expected = full_result.intra.u[chunk]
        differences.append((actual - expected).square().sum())
        expected_norm += expected.square().sum()
        actual_norm += actual.square().sum()
    if remainder:
        actual = prefix_result.intra.u[chunks, :remainder]
        expected = full_result.intra.u[chunks, :remainder]
        differences.append((actual - expected).square().sum())
        expected_norm += expected.square().sum()
        actual_norm += actual.square().sum()
    difference = torch.stack(differences).sum().sqrt().item()
    denominator = expected_norm.sqrt().item()
    return {
        "relative_frobenius": difference / max(denominator, 1e-300),
        "prefix_u_bitwise": difference == 0.0,
    }


def summarize_mask(mask: torch.Tensor, tiers: torch.Tensor) -> dict:
    return {
        "mask": mask.to(torch.int8).tolist(),
        "tiers": tiers.tolist(),
        "valid_events": int(mask.sum()),
        "keep_rate": float(mask.float().mean()),
    }


def run_aligned_case(prefix: int, key_dim: int, save_counterexample: Path | None) -> dict:
    suffix = 128
    full_x, full_y = make_shared_prefix_inputs(prefix, suffix, 20262000 + prefix + key_dim, key_dim)
    prefix_inputs = slice_inputs(full_x, 0, prefix)
    suffix_x = slice_inputs(full_x, prefix, prefix + suffix)
    suffix_y = slice_inputs(full_y, prefix, prefix + suffix)
    independent = run_reference(prefix_inputs, return_state_trace=True)
    one_x = run_reference(full_x, return_state_trace=True)
    one_y = run_reference(full_y, return_state_trace=True)
    prefix_chunks = prefix // 64
    independent_mask = independent.selector.mask
    mask_x = one_x.selector.mask[:, :prefix_chunks]
    mask_y = one_y.selector.mask[:, :prefix_chunks]
    state_x = one_x.state_trace[prefix_chunks]
    state_y = one_y.state_trace[prefix_chunks]
    weight = proxy_weight(prefix_inputs)
    private_slot = independent.final_state.clone()
    source_before = private_slot.clone()
    suffix_x.initial_state = private_slot
    suffix_y.initial_state = private_slot.clone()
    restored_x = run_reference(suffix_x)
    restored_y = run_reference(suffix_y)
    result = {
        "prefix_tokens": prefix,
        "suffix_tokens": suffix,
        "geometry": {"K": key_dim, "V": key_dim, "QK_heads": 16, "V_heads": 32},
        "prefix_mod_64": prefix % 64,
        "prefix_mod_128": prefix % 128,
        "selector": {
            "independent_P": summarize_mask(independent_mask, independent.selector.tiers),
            "P_plus_X_prefix": summarize_mask(mask_x, one_x.selector.tiers),
            "P_plus_Y_prefix": summarize_mask(mask_y, one_y.selector.tiers),
            "P_vs_PX_mask_differences": int((independent_mask != mask_x).sum()),
            "P_vs_PY_mask_differences": int((independent_mask != mask_y).sum()),
            "PX_vs_PY_prefix_mask_differences": int((mask_x != mask_y).sum()),
            "PX_vs_PY_tier_differences": int((one_x.selector.tiers != one_y.selector.tiers).sum()),
        },
        "U_prefix": compare_valid_u(independent, one_x, prefix),
        "state_at_P": {
            "independent_vs_PX": error_metrics(independent.final_state, state_x),
            "independent_vs_PY": error_metrics(independent.final_state, state_y),
            "PX_vs_PY": error_metrics(state_x, state_y),
        },
        "prefix_output": {
            "independent_vs_PX": error_metrics(independent.output, one_x.output[:prefix]),
            "independent_vs_PY": error_metrics(independent.output, one_y.output[:prefix]),
            "PX_vs_PY": error_metrics(one_x.output[:prefix], one_y.output[:prefix]),
        },
        "proxy_last_logits_at_P": {
            "independent_vs_PX": error_metrics(
                proxy_logits(independent.output, weight),
                proxy_logits(one_x.output[:prefix], weight),
            ),
            "independent_vs_PY": error_metrics(
                proxy_logits(independent.output, weight),
                proxy_logits(one_y.output[:prefix], weight),
            ),
            "note": "toy projection only; not full-model logits",
        },
        "private_slot_restore": {
            "source_state_unchanged": torch.equal(private_slot, source_before),
            "X_final_state_vs_one_shot": error_metrics(restored_x.final_state, one_x.final_state),
            "X_suffix_output_vs_one_shot": error_metrics(restored_x.output, one_x.output[prefix:]),
            "Y_final_state_vs_one_shot": error_metrics(restored_y.final_state, one_y.final_state),
            "policy": "independent-P state plus independently selected suffix; diagnostic nonzero-h0 extension",
        },
        "partitions": [],
    }
    for piece_size in (64, 128, 256, 1024, 2048):
        prefix_pieces = piece_lengths(prefix, piece_size)
        pieces = [*prefix_pieces, suffix]
        adaptive, adaptive_mask, adaptive_tiers = adaptive_partition_policy(full_x, pieces)
        frozen = run_frozen_partition(
            full_x,
            pieces,
            one_x.selector.mask,
            one_x.selector.tiers,
        )
        frozen_prefix_state = state_at_token(full_x, one_x, prefix)
        result["partitions"].append(
            {
                "scheduler_piece_tokens": piece_size,
                "piece_lengths": pieces,
                "adaptive_policy": {
                    "state_at_P_vs_independent": error_metrics(
                        adaptive.boundary_states[prefix], independent.final_state
                    ),
                    "state_at_P_vs_one_shot": error_metrics(adaptive.boundary_states[prefix], frozen_prefix_state),
                    "prefix_output_vs_independent": error_metrics(adaptive.output[:prefix], independent.output),
                    "final_state_vs_one_shot": error_metrics(adaptive.final_state, one_x.final_state),
                    "output_vs_one_shot": error_metrics(adaptive.output, one_x.output),
                    "mask_differences_vs_one_shot": int((adaptive_mask != one_x.selector.mask).sum()),
                    "piece_tiers": adaptive_tiers.tolist(),
                },
                "frozen_mask_policy": {
                    "state_at_P_vs_one_shot": error_metrics(frozen.boundary_states[prefix], frozen_prefix_state),
                    "final_state_vs_one_shot": error_metrics(frozen.final_state, one_x.final_state),
                    "output_vs_one_shot": error_metrics(frozen.output, one_x.output),
                    "corrected_U_vs_one_shot": error_metrics(frozen.corrected_u, one_x.corrected_u),
                    "output_bitwise": torch.equal(frozen.output, one_x.output),
                    "state_bitwise": torch.equal(frozen.final_state, one_x.final_state),
                },
            }
        )
    result["semantics"] = {
        "prefix_invariant": result["selector"]["PX_vs_PY_prefix_mask_differences"] == 0
        and result["prefix_output"]["PX_vs_PY"]["torch_equal"],
        "partition_invariant": all(
            item["adaptive_policy"]["mask_differences_vs_one_shot"] == 0
            and item["adaptive_policy"]["output_vs_one_shot"]["torch_equal"]
            for item in result["partitions"]
        ),
        "frozen_mask_partition_correct": all(
            item["frozen_mask_policy"]["output_bitwise"] and item["frozen_mask_policy"]["state_bitwise"]
            for item in result["partitions"]
        ),
        "ordinary_prefix_checkpoint_safe": restored_x.final_state.equal(one_x.final_state)
        and restored_y.final_state.equal(one_y.final_state),
    }
    if save_counterexample is not None:
        torch.save(
            {
                "contract": "same P, different X/Y; FP64 independent SparseGDN reference",
                "P": prefix_inputs,
                "X": suffix_x,
                "Y": suffix_y,
                "independent_P_output": independent.output,
                "independent_P_state": independent.final_state,
                "independent_P_mask": independent_mask,
                "P_plus_X": one_x.output,
                "P_plus_X_state_trace": one_x.state_trace,
                "P_plus_X_mask": one_x.selector.mask,
                "P_plus_Y": one_y.output,
                "P_plus_Y_state_trace": one_y.state_trace,
                "P_plus_Y_mask": one_y.selector.mask,
                "restored_X_output": restored_x.output,
                "restored_X_state": restored_x.final_state,
            },
            save_counterexample,
        )
    return result


def run_unaligned_case(prefix: int, key_dim: int) -> dict:
    suffix = 128
    full_x, full_y = make_shared_prefix_inputs(prefix, suffix, 20263000 + prefix + key_dim, key_dim)
    prefix_inputs = slice_inputs(full_x, 0, prefix)
    independent = run_reference(prefix_inputs, return_state_trace=True)
    one_x = run_reference(full_x, return_state_trace=True)
    one_y = run_reference(full_y, return_state_trace=True)
    state_x = state_at_token(full_x, one_x, prefix)
    state_y = state_at_token(full_y, one_y, prefix)
    try:
        adaptive_partition_policy(full_x, [prefix, suffix])
        partition_error = None
    except ValueError as error:
        partition_error = str(error)
    return {
        "prefix_tokens": prefix,
        "suffix_tokens": suffix,
        "geometry": {"K": key_dim, "V": key_dim, "QK_heads": 16, "V_heads": 32},
        "prefix_mod_64": prefix % 64,
        "prefix_mod_128": prefix % 128,
        "U_prefix": compare_valid_u(independent, one_x, prefix),
        "state_at_P": {
            "independent_vs_PX": error_metrics(independent.final_state, state_x),
            "independent_vs_PY": error_metrics(independent.final_state, state_y),
            "PX_vs_PY": error_metrics(state_x, state_y),
        },
        "prefix_output": {
            "independent_vs_PX": error_metrics(independent.output, one_x.output[:prefix]),
            "PX_vs_PY": error_metrics(one_x.output[:prefix], one_y.output[:prefix]),
        },
        "restore_status": "BLOCKED-CHUNK-ALIGNMENT",
        "partition_driver_status": partition_error or "UNEXPECTEDLY-ACCEPTED",
        "note": (
            "resetting the suffix to a fresh 64-token chunk would be a different policy; "
            "it is not used as a valid continuation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterexample", type=Path)
    parser.add_argument("--key-dim", type=int, default=8)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[128, 256, 1024])
    parser.add_argument("--extra-alignments", action="store_true")
    parser.add_argument("--unaligned", action="store_true")
    args = parser.parse_args()
    prefixes = list(args.prefixes)
    if args.extra_alignments:
        prefixes = [64, 192, *prefixes]
    aligned = []
    for index, prefix in enumerate(prefixes):
        counterexample = args.counterexample if index == 0 else None
        aligned.append(run_aligned_case(prefix, args.key_dim, counterexample))
        print(json.dumps({"completed_prefix": prefix, "K": args.key_dim}), flush=True)
    unaligned = []
    if args.unaligned:
        unaligned = [run_unaligned_case(prefix, args.key_dim) for prefix in (65, 127, 129)]
    report = {
        "schema_version": 1,
        "reference": "SparseGDN arXiv:2610.41420v1 independent FP64 implementation",
        "key_dim": args.key_dim,
        "state_dtype": "float64",
        "policy": {
            "one_shot": "paper global selector and group-2",
            "adaptive_partition": (
                "selector recomputed independently per scheduler piece; explicitly a partition-dependent policy"
            ),
            "frozen_partition": "same one-shot mask/tier; carries group-local even-chunk context across pieces",
            "restore": "diagnostic nonzero-h0 extension, not a paper compatibility claim",
        },
        "model_logits_status": "NOT-RUN; proxy projections in this file are not model logits",
        "aligned_cases": aligned,
        "unaligned_cases": unaligned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "aligned_cases": len(aligned),
                "unaligned_cases": len(unaligned),
                "prefix_invariant_cases": sum(case["semantics"]["prefix_invariant"] for case in aligned),
                "partition_invariant_cases": sum(case["semantics"]["partition_invariant"] for case in aligned),
                "frozen_partition_correct_cases": sum(
                    case["semantics"]["frozen_mask_partition_correct"] for case in aligned
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
