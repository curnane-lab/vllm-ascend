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
ran on two idle 910C chips with identical settings. Protocol: one shared
8,198-token prefix, unique suffixes, 32 greedy decode tokens, one round per
point. Aggregates:
`results/cb6242e/capacity9b_sweep_0930/aggregate.json` and
`results/cb6242e/capacity9b_sweep_dev11/aggregate.json`. Every completed row
has `restore_divergence_index=null` (token-exact prefix-cache restore).
Throughput includes prefill and scheduling.

| N | mode 0 tok/s (chip9 / chip11) | mode 3 tok/s (chip9 / chip11) | ratio |
|---:|---:|---:|---:|
| 16 | 44.30 / 44.26 | 14.46 / 14.44 | 0.327 / 0.326 |
| 32 | 56.23 / 57.74 | 23.97 / 24.06 | 0.426 / 0.417 |
| 64 | 66.71 / 65.75 | 34.93 / 34.77 | 0.524 / 0.529 |
| 96 | 70.10 / 70.72 | 41.18 / 40.70 | 0.587 / 0.576 |
| 128 | 72.34 / 72.75 | 45.37 / 45.65 | 0.627 / 0.628 |

The mode 3 ratio rises from 0.33× at N=16 to 0.63× at N=128. No pool or
scheduler cliff was reached by N=128. That 0.63× figure is not comparable
to the independent-prompt scan below: cache hits reduce prefill work only
for mode 0, and mode 3's chunked-prefill limitation blocks a symmetrical
shared-prefix setup above N=128. These rows stay as the token-exact restore
result. The earlier pre-fix eight-request run (`capacity9b_nocache/`) is a
failure record; its 8.5 tok/s mode 3 number is not comparable.

PIECEWISE with `mamba_block_size=128` and `block_size=128` was run at N=16.
Both modes completed and restore stayed token-exact. Throughput was
44.46 / 46.04 tok/s for mode 0 and 14.40 / 14.35 tok/s for mode 3. This
checks that the graph ran. It is not the unequal-page allocation matrix.

## Loader gate on `57ccb62` (910C, 2026-10-08)

A short mode 3 run of the symbol-skipping loader executed 672 flush calls.
Every `GetWorkspaceSize` returned 0, and the log has zero
`[gdn-lq-flush] slow-path exception` records. The binding logs the library
path it selected. Raw log:
`results/57ccb62/loader_gate_20261008_154748/`.

## 8,198-token independent-prompt scan (910C, 2026-10-08)

Each request is its own 8,198-token prompt, with prefix cache off.
Aggregate: `results/57ccb62/noprefix_capacity/aggregate.json`. Throughput is
total wall time, including prefill.

| N | mode 0 tok/s | mode 3 tok/s | ratio |
|---:|---:|---:|---:|
| 256 | 29.94 | 23.63 | 0.789 |
| 320 | 30.30 | 23.65 | 0.780 |
| 384 | 30.03 | 23.95 | 0.797 |
| 448 | 30.19 | 23.73 | 0.786 |
| 512 | 30.46 | 23.92 | 0.785 |

No admission or pool cliff appears through N=512. This vLLM build exposes
no engine metric for the admitted block count, so the remaining pool
headroom is unmeasured. The ratio stays inside 0.780–0.797 while N doubles.
The paper-scale 1.4× claim is still open.

`results/57ccb62/admission_noprefix2/` and
`results/57ccb62/admission_final/` do not vary page sizes. The
`mamba_block_size=128`, `block_size=128` run remains a PIECEWISE graph
smoke.

## Production decode check (910C, 2026-10-08)

The production path has no step-versus-flush timer, so this run does not
split Triton step time from flush v1. Mode 3 logged zero slow-path
exceptions. On the 64-token greedy prompts, prompts 0 and 1 match mode 0
for every token. Prompts 2 and 3 diverge in the later tokens at a near tie.
Raw files: `results/57ccb62/decode_segment/`. This is separate from the
768-token identical run and from the token-27 Torch-versus-mode-3 split.
Summary of the night: `results/57ccb62/night_report.md`.

## Direct-readout step (910C, commit base `7967d96`)

`VLLM_ASCEND_GDN_LQ_DIRECT_READOUT=1` is the opt-in path. The host kernel
listing does not match the materializing step: the compensator terms are
absent, `s_v` is divided by 127 a second time, the 16-row partials are not
scattered back onto the value axis, and the shared epilogue reads `w`
without loading it on that path. `tests/test_step_triton.py` never sets
the flag, so its 0.004242 relative error is the materializing kernel
against Torch, inside the `2e-2` gate. It is not a direct-readout result.

`scripts/bench_step_mode.py` sets the flag, then times a Python loop of 96
steps. Each iteration draws new inputs, normalizes them, runs the step,
and calls AscendC flush, including on steps whose window is not full.
Shape: B=8, H=32, T=96, 50 iterations. Pools are reset between iterations.
The reported medians are 177.771 ms (materializing) and 178.268 ms
(direct), a 0.50 ms gap, 0.28%. That gap is on the whole loop. Input
generation sits inside the timer, and the script does not compare outputs.
Raw files: `results/7967d96/step_mode_bench_dense.jsonl`,
`results/7967d96/step_mode_bench_direct.jsonl`, and
`results/7967d96/step_mode_bench.md`. The kernel and the script are on the
910C host and are not in this tree. Leave the flag off.

## Static pool budget on vllm-ascend 0.27.1 (910C, 2026-10-09)

This run is a port onto official vllm 0.27.1 + vllm-ascend 0.27.1rc1
(`3b3188623`), CANN 9.1.0, torch 2.10.0 / torch_npu 2.10.0.post4,
triton-ascend 3.2.2. Device: Ascend910_9382, 64GB. Weights:
`/mnt/weight/Qwen3.5-9B`. The patch lives in that container's
`/vllm-workspace/vllm-ascend` tree. It is not in this repository. The
binding replaces `fla_npu` with a local ctypes runtime.

Startup log only, no generation. Same `max_model_len=8192`,
`gpu_memory_utilization=0.85`, prefix caching on, eager.

| geometry | attn block | mamba page | GPU KV tokens | concurrency @8192 | vs mode 0 | outputs |
|---|---:|---:|---:|---:|---:|---|
| mode 0 dense | 1024 | 4,243,456 | 595,675 | 72.71× | 1.00× | matches |
| mode 3 pad, production default | 512 | 2,146,304 | 749,940 | 91.55× | 1.26× | passes |
| mode 3 compact | 256 | 1,097,728 | 848,949 | 103.63× | 1.425× | cold start broken |
| mode 3 block 256 + pad page | 256 | 2,146,304 | 434,176 | 53.00× | 0.73× | passes, fewer tokens |

91.55 / 72.71 = 1.26. 103.63 / 72.71 = 1.425. 53.00 / 72.71 = 0.73. The
1.425× row is void: compact pages overwrite attention V. The old
0.93–1.0× band was the vendored fork's 384-vs-640 block account, not this
page geometry.

Compact stays broken with prefix caching off (divergence at token 0 / 2).
Mode 0 restore, mode 3 pad restore, and block-256-plus-pad restore are 4/4.
Forcing block 256 is not the failure. The shared KV tensor is
`[conv 49,152 B | K | V]`. The ssm view is a dense slice of stride
1,048,576 B from the end of conv, and it stays clear of V only while

```text
page >= 256 * 4096 + 49,152 + 1,048,576 = 2,146,304 B
```

The compact page is 1,097,728 B, so high slot ids land in V. That is a
layout collision, not a restore bug and not a tunable page size. On this
stack the padded 1.26× concurrency is the structural maximum. A separate
mamba tensor would be required to chase the void 1.425× figure.

Functional checks on the same port, weights read-only:

- mode 1, 48 tokens: 3/4, prompt 3 diverges at token 36.
- mode 2 matches mode 1 bitwise.
- mode 3, 48 tokens: 4/4 against mode 0, zero `[gdn-lq-flush] slow-path`,
  library path logged.
- 768 tokens, mode 0 vs mode 3: divergences at tokens 101 and 132, ratio
  0.77×. This is a different stack from the vendored-fork 768-token
  identical run.
- 64-token top-2 gaps: mode 0 vs mode 3 identical on all 4 prompts. Literal
  gap 0.0000 ties on prompts 1 and 3 did not flip. The vendored-fork
  prompt 2/3 late split was not reproduced here.

`VLLM_ASCEND_GDN_LQ_COMPACT_PAGE` stays off. `VLLM_ASCEND_GDN_LQ_STEP_V2`
stays off.

## Still open

- On the 0.27.1 shared KV layout, a compact mamba page cannot be correct.
  The padded geometry already measures 1.26× concurrency at a fixed 8192
  denominator. Reaching the void 1.425× number requires a separate mamba
  allocation, not another page-size flag or a larger N.
- FULL graph mode remains unsupported by the Ascend GDN backend.
- The 0.780–0.797× end-to-end ratio is still unattributed. The B=8
  direct-readout timing includes input generation and a flush call on
  every step, and it does not check outputs. The next measurement is the
  prefill split: reconstruct, quantize, Triton step, and flush v1 on one
  8,198-token run. A step-only timing, with inputs prepared outside the
  timer, waits on a direct-readout that matches the materializing formula.
  t3 remains the first v2 kernel target
  if that benchmark line continues, with a w/p/fill and decoded-state
  regression, then t1/t2, then fprep. It does not move this ratio.
  Production flags stay off.
- The token-27 evidence points at accumulated Torch quantization error.
  A one-layer replay is optional. The new 64-token near-ties are prompts
  2 and 3 only.
- Task-level accuracy is still open. The completed AB slice is 20 prompts
  per dataset; the full five-dataset sweep is 14,634 prompts.
- `scripts/bench_capacity_graph.py` is the overnight driver. It accepts
  `--block-size`, `--no-chunked-prefill`, and an external NPU snapshot.
  That file is on the 910C host and is not in this tree.
  `scripts/bench_capacity.py` remains the eager shared-prefix script.
  Loader path selection and `smoke_qwen35.py --block-size` are already on
  `main`.
