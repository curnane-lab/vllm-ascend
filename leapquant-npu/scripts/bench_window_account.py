import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools, WINDOW
from vllm_ascend.ops.gdn_lq_flush_ascendc import flush_pools_ascendc
from vllm_ascend.ops.triton.gdn_leapquant_step import gdn_leapquant_step


def _synchronized_ms(fn):
    torch.npu.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.npu.synchronize()
    return result, (time.perf_counter() - started) * 1e3


def _median(values):
    return statistics.median(values) if values else 0.0


def _make_inputs(batch, heads):
    torch.manual_seed(321)
    dense = torch.randn(batch, heads, 128, 128, device="npu") * 0.02
    q = F.normalize(torch.randn(batch, heads, 128, device="npu"), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(batch, heads, 128, device="npu"), dim=-1).to(torch.bfloat16)
    v = torch.randn(batch, heads, 128, device="npu").to(torch.bfloat16)
    g = torch.full((batch, heads), -0.05, device="npu")
    beta = torch.full((batch, heads), 0.3, device="npu").to(torch.bfloat16)
    idx = torch.arange(batch, device="npu", dtype=torch.int32)
    actual = torch.ones(batch, dtype=torch.int32, device="npu")
    return dense, q, k, v, g, beta, idx, actual


def _run_window(mode, pools, dense_state, inputs):
    _, q, k, v, g, beta, idx, actual = inputs
    account = {
        "window_steps": WINDOW,
        "step_calls": 0,
        "step_ms": 0.0,
        "flush_calls": 0,
        "effective_flush_calls": 0,
        "due_instances": 0,
        "flush_ms": 0.0,
        "flush_backends": {},
    }

    def dense_steps():
        nonlocal account
        for _ in range(WINDOW):
            _, elapsed = _synchronized_ms(
                lambda: recurrent_gated_delta_rule(
                    q,
                    k,
                    v,
                    dense_state,
                    g=g,
                    beta=beta,
                    scale=128**-0.5,
                    actual_seq_lengths=actual,
                    ssm_state_indices=idx,
                )
            )
            account["step_calls"] += 1
            account["step_ms"] += elapsed

    def lq_steps():
        nonlocal account
        long_idx = idx.long()
        for _ in range(WINDOW):
            if mode == 1:
                step = lambda: pools.decode_step(long_idx, k, q, v, g, beta, 128**-0.5)
            else:
                step = lambda: gdn_leapquant_step(pools, long_idx, k, q, v, g, beta, 128**-0.5)
            _, elapsed = _synchronized_ms(step)
            account["step_calls"] += 1
            account["step_ms"] += elapsed

            due = int((pools.fill[long_idx] >= WINDOW).sum().item())
            account["flush_calls"] += 1
            account["effective_flush_calls"] += int(due > 0)
            account["due_instances"] += due
            if mode == 3:
                ok, elapsed = _synchronized_ms(lambda: flush_pools_ascendc(pools, long_idx))
                backend = "ascendc" if ok else "torch_fallback"
                if not ok:
                    _, torch_elapsed = _synchronized_ms(lambda: pools.flush(long_idx))
                    elapsed += torch_elapsed
            else:
                backend = "torch"
                _, elapsed = _synchronized_ms(lambda: pools.flush(long_idx))
            account["flush_ms"] += elapsed
            account["flush_backends"][backend] = account["flush_backends"].get(backend, 0) + 1

    _, window_ms = _synchronized_ms(dense_steps if mode == 0 else lq_steps)
    account["window_ms"] = window_ms
    account["other_ms"] = window_ms - account["step_ms"] - account["flush_ms"]
    return account


def _summarize(iterations):
    keys = (
        "window_ms",
        "step_ms",
        "flush_ms",
        "other_ms",
        "step_calls",
        "flush_calls",
        "effective_flush_calls",
        "due_instances",
    )
    return {key: _median([row[key] for row in iterations]) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", type=Path, default=_REPO / "leapquant-npu/runs/window_account.json")
    args = parser.parse_args()
    if any(mode not in (0, 1, 2, 3) for mode in args.modes):
        parser.error("modes must be selected from 0, 1, 2, 3")

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
    payload = {
        "schema_version": 1,
        "git_sha": git_sha,
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=_REPO, text=True).strip()
        ),
        "config": {
            "batches": args.batches,
            "modes": args.modes,
            "heads": args.heads,
            "window_steps": WINDOW,
            "warmup": args.warmup,
            "iters": args.iters,
            "flush_semantics": "production: flush after every decode step",
            "mode_semantics": {
                "0": "dense recurrent step",
                "1": "Torch step + Torch flush",
                "2": "K-space Triton step + Torch flush",
                "3": "K-space Triton step + old AscendC flush, Torch fallback",
            },
            "timing": "synchronized wall time per step and flush; other_ms includes timing and accounting overhead",
        },
        "device": {
            "name": torch.npu.get_device_name(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "runs": [],
        "comparisons": [],
    }

    for batch in args.batches:
        inputs = _make_inputs(batch, args.heads)
        medians = {}
        for mode in args.modes:
            dense_state = inputs[0].clone()
            pools = None
            if mode:
                lq_state = inputs[0].clone()
                pools = GDNLeapQuantPools(lq_state)
                pools.init_from_dense(lq_state, inputs[6].long())
            for _ in range(args.warmup):
                _run_window(mode, pools, dense_state, inputs)
            iterations = [_run_window(mode, pools, dense_state, inputs) for _ in range(args.iters)]
            median = _summarize(iterations)
            medians[mode] = median
            payload["runs"].append(
                {
                    "batch": batch,
                    "mode": mode,
                    "iterations": iterations,
                    "median": median,
                }
            )
            print(
                f"B={batch} mode={mode} window={median['window_ms']:.3f} ms "
                f"step={median['step_ms']:.3f} ms flush={median['flush_ms']:.3f} ms "
                f"other={median['other_ms']:.3f} ms flush_calls={median['flush_calls']:.0f} "
                f"effective={median['effective_flush_calls']:.0f} due={median['due_instances']:.0f}",
                flush=True,
            )
        if 0 in medians:
            for mode, median in medians.items():
                if mode:
                    comparison = {
                        "batch": batch,
                        "mode": mode,
                        "dense_window_ms": medians[0]["window_ms"],
                        "leapquant_window_ms": median["window_ms"],
                        "speedup_vs_dense": medians[0]["window_ms"] / median["window_ms"],
                        "target_window_ms_at_1_10x": medians[0]["window_ms"] / 1.10,
                    }
                    payload["comparisons"].append(comparison)
                    print(
                        f"B={batch} mode={mode} speedup_vs_dense={comparison['speedup_vs_dense']:.3f}x",
                        flush=True,
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
