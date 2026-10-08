# M5 measurement update (910C reproduction, 2026-10-07)

Upstream base: `566bf15`. Device: Ascend 910C. Production flags stayed off
(`VLLM_ASCEND_GDN_LQ_STEP_V2=0`, flush v2 not wired into `gdn.py`).

## What this file is

The 9B restore trajectories cited by `M5-capacity.md` (`runs/m5_restore.jsonl`,
`runs/smoke_t2.jsonl`) were not in `566bf15`. `runs/m5_restore.jsonl` in this
commit is a four-row reconstruction of that report, not a per-token dump.
A byte-for-byte audit of the 9B compact-vs-dense or hot-vs-cold claims still
requires rerunning them on the 9B checkpoint.

Fresh NPU rows below were measured on the 910C machine. The Qwen3.5-9B
checkpoint used by the original reports was not on that machine.

## Review fixes

- `tests/test_flush_v2.py`: header matches `ABS_GATE=0.005`. The ratio stays informational.
- `gdn_lq_flush_v2.py:reset_due_gate(dev)`: deletes every mirror whose key starts with `str(dev)`.
- `tests/test_due_gate.py:test_mixed_due_batch`: after step 15 has skipped, one `st["fills"]` record is set to `(WINDOW, 15)` and probed again at step 15. Step 16 is already due for every slot in this rhythm, and `(WINDOW, original_observed_step)` takes the post-fold credit path, so neither of those probes tests the injection. This corrected assertion has not been rerun on 910C.
- Import and vendor paths are repo-relative. Model checkpoints resolve through `LQ_MODEL_PATH`, falling back to `/mnt/workspace/models/Qwen3.5-9B` so the original bench host keeps working.

## Protocol rows reconstructed from the 9B report

`runs/m5_restore.jsonl`:

- mode 3 hot-vs-cold restore: 3/4 token-exact, prompt 0 common prefix 45/48.
- chunked mode 3 restore: byte-identical to the unchunked mode 3 run.
- compact pool vs dense pool decode: 4/4 token-identical. Raw files absent.
- mode 0 hot-vs-cold restore: 2/4 token-exact, prompt 1 at token 36, prompt 2 at token 0. Pre-existing on that stack.

## Fresh smoke (Qwen3.5-4B, NPU 0, greedy, 48 tokens)

Raw files were left on the 910C host at
`results/m5_capacity/20261007_104321/{mode0_mode0,mode3_mode3}.jsonl`.
They are not in this commit.

```text
mode 0 vs mode 3:
prompt 0: common-prefix=48, IDENTICAL
prompt 1: common-prefix=48, IDENTICAL
prompt 2: common-prefix=48, IDENTICAL
prompt 3: common-prefix=48, IDENTICAL
```

This is a 4B sanity trace. It does not replace the 9B mode-3 result
(3/4, prompt 3 diverges at token 36).

## Fresh 768-token mode 3 trace

Raw files were left on the 910C host at
`results/m5_capacity/accuracy_20261007_104805/{mode0_768,mode3_768}.jsonl`.
They are not in this commit. Both sides used the default production path.
The 9B checkpoint was unavailable on that host; this trace belongs with the
4B smoke above, not with the 9B token-36 result.

```text
prompt 0: first divergence at token 221 of 768
prompt 1: first divergence at token 79 of 768
```

Both prompts generated all 768 tokens. The log had neither
`[gdn-lq-epilogue] slow-path exception` nor `[lq-v2] ... unavailable`.
Free greedy leaves the reference well before 768 tokens. The teacher-forced
checkpoint curve, which saturates near 1.5% by step 1072, does not describe
this trajectory.

## Chain event profile (910C)

Script: `tests/bench_flush_v2_chain_events.py`. One `_chain7` enqueue, a
device event between stages, no per-stage `synchronize`. fprep is not in
`_chain7` and is not in this table. The raw jsonl stayed on the 910C host
(`results/m5_upstream/chain_20261007_120411/chain_events.jsonl`).

| B | g1 | g2 | t1 | g3 | t2 | g4 | t3 | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.049 | 0.057 | 0.177 | 0.019 | 0.170 | 0.019 | 0.687 | 1.27 ms |
| 64 | 0.311 | 0.096 | 0.672 | 0.054 | 0.623 | 0.050 | 2.666 | 4.59 ms |
| 256 | 0.944 | 0.352 | 2.704 | 0.431 | 2.872 | 0.408 | 10.284 | 18.13 ms |

B=256 stage sum is 18.00 ms against a 18.13 ms wall. t3 is 57% of the chain.
t1+t2 is 5.58 ms and scales with batch. The four GEMMs sum to 2.14 ms.
These numbers are the 910C baseline. The 910B chain wall of 33.2 ms and the
per-stage-sync sum of 38.4 ms stay on that other device.

## Still open

- Rerun `test_mixed_due_batch` on 910C after the step-15 injection.
- 8K+ / shared-prefix long-tail capacity, and FULL/PIECEWISE with unequal pages.
- Mode 3 vs Torch mode 1 vs fp32 dense at the first 768-token divergence (token 79).
- t3 remains the first kernel target if flush work continues, with a w/p/fill and decoded-state regression. t1/t2 are the next AIV slice. Production flags stay off.
