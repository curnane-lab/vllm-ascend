# M5 measurement update (910C reproduction, 2026-10-08)

Initial upstream base: `566bf15`; later rows include merged `cb6242e`.
Device: Ascend 910C. Production flags stayed off
(`VLLM_ASCEND_GDN_LQ_STEP_V2=0`, flush v2 not wired into `gdn.py`).

## What this file is

The 9B restore trajectories cited by `M5-capacity.md` (`runs/m5_restore.jsonl`,
`runs/smoke_t2.jsonl`) were not in `566bf15`. `runs/m5_restore.jsonl` in that
commit is a four-row reconstruction of that report, not a per-token dump.
A byte-for-byte audit of the 9B compact-vs-dense or hot-vs-cold claims still
requires rerunning them with the same protocol on the 9B checkpoint.

Fresh NPU rows below were measured on the 910C machine. The Qwen3.5-9B
checkpoint is available at `/data1/models/Qwen3.5-9B`; later sections use it.

## Review fixes

- `tests/test_flush_v2.py`: header matches `ABS_GATE=0.005`. The ratio stays informational.
- `gdn_lq_flush_v2.py:reset_due_gate(dev)`: deletes every mirror whose key starts with `str(dev)`.
- `tests/test_due_gate.py:test_mixed_due_batch`: after step 15 has skipped, one `st["fills"]` record is set to `(WINDOW, 15)` and probed again at step 15. The corrected assertion has been rerun on 910C; see the due-gate section below.
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

## Fresh 768-token mode 3 trace (Qwen3.5-4B)

Raw files were left on the 910C host at
`results/m5_capacity/accuracy_20261007_104805/{mode0_768,mode3_768}.jsonl`.
They are not in this commit. Both sides used the default production path.
This 4B trace is separate from the 9B token-27 classification below.
Generated-token index 79 in this file is not token 27.

```text
prompt 0: first divergence at generated-token index 221 of 768
prompt 1: first divergence at generated-token index 79 of 768
```

Both prompts generated all 768 tokens. The log had neither
`[gdn-lq-epilogue] slow-path exception` nor `[lq-v2] ... unavailable`.
Free greedy leaves the reference well before 768 tokens. The teacher-forced
checkpoint curve, which saturates near 1.5% by step 1072, does not describe
this trajectory.

## Due gate on merged `cb6242e` (910C)

`results/cb6242e/due_gate_20261008_001319/` contains the full 12-item matrix
run on chip 0 at commit `cb6242e`. It includes the corrected mixed-due
assertion and the tuple-key reset test. All items PASS; the nothing-due
median is 0.0914 ms.

## Chain event profile (910C)

Script: `tests/bench_flush_v2_chain_events.py`. fprep is enqueued on the same
stream before the `_chain7` stages; a device event separates each stage
without per-stage host synchronization. Raw files:
`results/cb6242e/fprep_chain_20261008_031416/chain_events.jsonl`. The
earlier seven-stage table remains in
`results/m5_upstream/chain_20261007_120411/chain_events.jsonl`.

| B | fprep | g1 | g2 | t1 | g3 | t2 | g4 | t3 | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.162 | 0.028 | 0.030 | 0.170 | 0.019 | 0.170 | 0.019 | 0.687 | 1.44 ms |
| 64 | 0.578 | 0.311 | 0.096 | 0.672 | 0.054 | 0.623 | 0.050 | 2.666 | 5.09 ms |
| 256 | 2.173 | 1.018 | 0.348 | 2.589 | 0.448 | 2.589 | 0.434 | 10.639 | 20.36 ms |

B=256 stage sum is 20.24 ms against a 20.36 ms wall. t3 is 52% of the
stages. t1+t2 is 5.18 ms and scales with batch. The four GEMMs sum to
2.25 ms. fprep is 2.17 ms, below t2. These numbers are the 910C baseline.
The 910B chain wall of 33.2 ms and the per-stage-sync sum of 38.4 ms stay
on that other device.

## Token-27 three-path classification (Qwen3.5-9B, 910C, 2026-10-08)

Upstream base: `cb6242e`. Raw traces are
`results/cb6242e/token79_20261008_030718/{mode0,mode1,mode3}_trace.jsonl`.
One greedy prompt generated 28 tokens (`target=27`). All three modes shared
tokens 0..26. At token index 27:

| path | argmax |
|---|---:|
| mode 0, fp32 dense | 103808 |
| mode 1, Torch LeapQuant | 98153 |
| mode 3, production Triton step + flush v1 | 103808 |

Mode 1 leaves mode 0 first; mode 3 still matches mode 0 at the split. This
classifies the split as the Torch quantization path, not the kernelized mode-3
path. The logs had neither `[gdn-lq-epilogue] slow-path exception` nor
`[lq-v2] ... unavailable`.

The same prompt was then rerun with dense and Torch LeapQuant on chips 0 and 1.
Raw files are `results/cb6242e/layer_trace_20261008_063433/` and
`results/cb6242e/layer_trace_20261008_063420/`. The 24 GDN layers at step 26
have no argmax mismatch; their outputs differ by at most 0.00927 relative
L2 (`layer_output_relative_error.json`). The token-27 split is an accumulated
quantization trajectory, not one GDN layer whose argmax flipped. This is
separate from the 910B/9B token-36 observation.

## 9B mode 0 vs mode 3 over 768 tokens (910C, 2026-10-08)

The first run (`results/cb6242e/mode03_768_9b_20261008_083243/`) was
token-identical, but its mode 3 log had 18,408 AscendC flush-v1 slow-path
errors and therefore used the Torch fallback after every fold. It records
the production sampling trajectory only.

Root cause: `gdn_lq_flush_ascendc._binding` loaded an older
`libcust_opapi.so` that did not export `aclnnGdnLqFlushGetWorkspaceSize`,
so every e2e `GetWorkspaceSize` returned `161001`. The compact e2e layout
was `state_shape=(slots,32,8192)`, `slot_stride_bytes=1048576`,
`num_heads=32`, `idx_shape=(1,)`. That layout is not what rejected the call.
The binding now skips any library that does not export the symbol.

After the run selected the FLA NPU package that exports the symbol, a
4-token probe had zero slow-path errors. The gate rerun is
`results/cb6242e/mode03_768_9b_zeroflush/`. Raw files:
`{mode0,mode3}_768.jsonl`. Both use `/data1/models/Qwen3.5-9B`, greedy,
`block_size=128`, no prefix cache, and the production mode 3 path.
Mode 0 and mode 3 are token-identical through all 768 generated tokens and
the mode 3 log has zero `[gdn-lq-flush] slow-path exception` records.

## 9B shared-prefix 8K capacity sweep (910C, 2026-10-08)

After that run selected the library that exports the symbol, a second sweep
ran on two idle 910C chips with identical settings. Protocol: one shared 8,198-token prefix, unique suffixes,
32 greedy decode tokens, one round per point. Raw files and aggregates:
`results/cb6242e/capacity9b_sweep_0930/` and
`results/cb6242e/capacity9b_sweep_dev11/`. All completed rows have
token-exact prefix-cache restore. Throughput includes prefill and scheduling.

| N | mode 0 tok/s (chip9 / chip11) | mode 3 tok/s (chip9 / chip11) | ratio |
|---:|---:|---:|---:|
| 16 | 44.30 / 44.26 | 14.46 / 14.44 | 0.327 / 0.326 |
| 32 | 56.23 / 57.74 | 23.97 / 24.06 | 0.426 / 0.417 |
| 64 | 66.71 / 65.75 | 34.93 / 34.77 | 0.524 / 0.529 |
| 96 | 70.10 / 70.72 | 41.18 / 40.70 | 0.587 / 0.576 |
| 128 | 72.34 / 72.75 | 45.37 / 45.65 | 0.627 / 0.628 |

The mode 3 ratio rises from 0.33× at N=16 to 0.63× at N=128. No pool or
scheduler cliff was reached by N=128. The earlier pre-fix eight-request run
(`capacity9b_nocache/`) is a failure record; its 8.5 tok/s mode 3 number is
not comparable.

PIECEWISE with `mamba_block_size=128` and `block_size=128` was run at N=16.
Both modes completed and restore stayed token-exact. Throughput was
44.46 / 46.04 tok/s for mode 0 and 14.40 / 14.35 tok/s for mode 3. This
checks that the graph ran. It is not the unequal-page allocation matrix.

## Still open

- The 8K sweep stops at N=128 without a cliff. The paper-scale 1.4× claim
  still needs a larger N and a true unequal-page allocation matrix.
- FULL graph mode remains unsupported by the Ascend GDN backend.
- The token-27 evidence points at accumulated Torch quantization error.
  A one-layer replay is optional.
- Task-level accuracy is still open. The completed AB slice is 20 prompts
  per dataset; the full five-dataset sweep is 14,634 prompts.
- t3 remains the first kernel target if flush work continues, with a w/p/fill
  and decoded-state regression. t1/t2 are the next AIV slice, then fprep.
  Production flags stay off.
