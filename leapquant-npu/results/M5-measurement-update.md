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
layout collision, not a restore bug and not a tunable page size. The
1.26× figure is the startup KV-token budget under prefix-cache align
(KVCacheManager page account plus the vLLM concurrency formula), not a
live served concurrency. Among the shared-layout geometries that were
measured and that kept outputs correct, the padded page is the largest.
That is a page-correctness bound. It does not close a serving
measurement, and a later ramp can still lose the budget to execution
cost. A separate mamba tensor would be required to go past this page.
After that split the overhead has to be counted again; the void 1.425×
row stays void. The no-prefix-cache startup budget and the live 96-way
ramp are below.

Functional checks on the same port, weights read-only:

- mode 1, 48 tokens: 3/4, prompt 3 diverges at token 36.
- mode 2 matches mode 1 bitwise.
- mode 3, 48 tokens: 4/4 against mode 0, zero `[gdn-lq-flush] slow-path`,
  library path logged.
- 768 tokens, mode 0 vs mode 3: divergences at tokens 101 and 132, ratio
  0.77×. This is a different stack from the vendored-fork 768-token
  identical run. The flips are compatible with accumulated quantization
  error. An implementation difference on this port is not ruled out.
- 64-token top-2 gaps: mode 0 vs mode 3 identical on all 4 prompts. Literal
  gap 0.0000 ties on prompts 1 and 3 did not flip. The vendored-fork
  prompt 2/3 late split was not reproduced here.

Parity on the same container, four chips, six tests:

| check | result |
|---|---|
| `test_pools_parity.py` | pass |
| `test_step_triton.py` | pass |
| `test_flush_ascendc.py` | pass |
| `test_flush_strided.py` | smoke pass |
| `test_step_ascendc.py` | pass; 96-step output rel ≤ 0.0095, end-of-window state ≤ 0.0102 |
| `test_step_v2.py` | pass on one pool; `w`/`p`/`fill` bit-exact there, end-of-window state ≤ 0.0104 (overall mean-abs, not per-head relative L2) |

The reported step-v2 times 0.408 / 0.612 / 1.752 ms at B=16/64/256 are a
step-plus-reset loop wall on this container. The loop includes
`fill`/`w.zero_()`. They are not pure 3-op kernel times, and they do not
replace the vendored-fork M2 table. The bit-exact `w`/`p`/`fill` row is
the single-pool test. It does not cover the 24-pool alternation below.
The harness edits that make the tests import
`gdn_lq_aclnn_runtime` instead of `fla_npu`, the `libcust_opapi.so`
symlink, and the mirror-package rename stay on that host.

`VLLM_ASCEND_GDN_LQ_COMPACT_PAGE` stays off. `VLLM_ASCEND_GDN_LQ_STEP_V2`
stays off. Flush v2's seven-stage chain is still not wired into `gdn.py`.
The container's `patches/vllm-ascend-0271/` tree is not in this repository.

## Production timing on the 0.27.1 port (eager, B=2, 2×768 tokens)

`VLLM_ASCEND_GDN_LQ_TIMER=1` is an env-gated NPU-event timer added to
`gdn.py` on that container. The timer is not in this tree. Wall: mode 0
41.9 s, mode 3 54.2 s, ratio 0.773×, matching the untuned 0.77× baseline.
Timer overhead is approximately symmetric between the two modes.

Averages are per layer-step and already include B=2. The extrapolation is
768 tokens × 24 GDN layers = 18,432 layer-steps. Prefill counts are call
counts, not layer-steps.

| region | avg | extrapolated |
|---|---:|---:|
| dense decode recur (mode 0) | 0.103 ms / layer-step | 1.90 s |
| lq step, Triton (mode 3) | 0.208 ms / layer-step | 3.83 s |
| lq flush v1, AscendC (mode 3) | 0.344 ms / layer-step | 6.35 s |
| mode 3 decode total | 0.552 ms / layer-step (5.4× dense) | 10.18 s |
| prefill quantize | 10.72 ms × 48 calls | 0.51 s |
| prefill reconstruct | 4.46 ms × 48 calls | 0.21 s |

The wall gap is 54.2 − 41.9 = 12.3 s. Recomputed from the reported
averages over 18,432 layer-steps, step is only the increment over dense:

| slice | extra vs dense | share of 12.3 s |
|---|---:|---:|
| step `(0.208 − 0.103) × 18432` | 1.935 s | 15.7% |
| flush `0.344 × 18432` | 6.341 s | 51.6% |
| prefill `10.72×48 + 4.46×48` ms | 0.729 s | 5.9% |
| unattributed residual | 3.295 s | 26.8% |

The full 3.83 s step extrapolation includes the 1.90 s dense already
spent on the same layer-steps. The 3.295 s remainder is an unattributed
residual. Mode 3 does add two GQA `repeat_interleave`s and one flush
launch per layer-step, and that host path has no timer of its own.
An NPU-event interval can include the device waiting for the next host
submit, so these averages are interval times, not isolated kernel times.

The timer reports every 4096 regions. The written record has the
averages only, not the per-region counts, and it does not confirm that
the last partial bucket was included. B=2 ran inside each call, so the
18,432 count is not multiplied by 2 again. Raw logs stay on the container.

The flush timer region is the largest measured slice of the gap. Removing
the entire reported 6.35 s would leave mode 3 at 54.2 − 6.35 = 47.85 s,
which is 41.9 / 47.85 ≈ 0.876× dense. On this B=2 wall, flush alone does
not pass dense. That 0.876× figure stays on this wall. It is not an input
to a high-batch capacity model.

The 0.344 ms average also mixes every call. It does not separate a
nothing-due launch, a partial fold, an all-due fold, or a cold bind.
The next split, on a synchronized window, records for nothing-due: call
count, host submit time, device time; for mixed and all-due: due-slot
count, call count, device time; for the window: 16-step total, fold
count, and a state check. A controlled upper bound calls flush only when
a slot is known due. Production still has to cover async requests, slot
recycle, and restore, so a global step modulo is not that bound.
Flush v2 t3 stays a candidate on the experimental chain. Whether a
replacement is worth shipping is the full-window cost in the operator
gate below, not a further pass of vector-only tuning.

## Live 96-way ramp (no prefix cache, eager, two chips)

96 requests were submitted as 8000 prompt tokens plus 256 decode
tokens. The pasted note does not record `max_model_len` or the tokenized
prompt length. If the run kept `max_model_len=8192`, then 8000 + 256 =
8256 exceeds that cap, and the real prompt length is unknown. Startup
concurrency on this no-cache page geometry is mode 0 92.55× and mode 3
106.00×, which is 1.145× (+14.5%). That geometry is separate from the
align-mode 1.26× budget above.

| | wall | generated tok/s | peak running |
|---|---:|---:|---:|
| mode 0 | 72.8 s | 247.9 | ~92 |
| mode 3 | 134.6 s | 134.1 | 96 |
| ratio | 0.54× | | |

Both runs report zero preemption and that all 96 requests were admitted.
Admission is not simultaneous residency. Mode 0's peak running count is
about 92, so some requests can be waiting. Zero preemption does not
establish an empty queue.

96 × 256 = 24,576 output tokens. tok/s × wall is 247.9 × 72.8 ≈ 18,047
on mode 0 and 134.1 × 134.6 ≈ 18,050 on mode 3. The two products agree
with each other and disagree with 24,576. The note does not say whether
that gap is early EOS, a different time denominator, or a different
completion count. Until the completed token count is in the log, 0.54×
is the ratio of the reported rates on this decode-heavy submission, not
a 256-token-per-request result.

On this submission the budget did not show up as throughput. Whether the
budget pays under an admission-bound, light-decode load is still a
hypothesis. Short generations can lose it to prefill reconstruct and
quantize. The next capacity points, if that hypothesis is tested, use
one cache and page configuration, sit between the two admission
boundaries, and record waiting, running, TTFT, TPOT, completions, and
output tokens. N above 106 is one such point, not a requirement.
1.26× stays the align startup budget. 0.54× stays this ramp's rate ratio.
Multiplying that rate ratio by 1.145 or 1.26 counts concurrency twice.
The break-even that replaces that product is in the section below.

## Step v2, 24 pools (container test, flag stays off)

`tests/test_step_v2_multipool.py` is on the container, not in this tree.
Log: `/tmp/opencode/parity/test_step_v2_multipool_chip2.log`. Phase A is
a pure-Torch 24-pool reference. Phase B replays the same seed through
the v2 fused entry and AscendC flush.

1152 fused calls across 24 same-shape, different-address pools,
alternating layers, reported 0 executor fallbacks. Zero fallbacks means
those calls did not take the fallback path. It does not show that the
executor bound the address of the pool being written. A successful call
on the wrong address also reports no fallback. The old E1 failure is
not shown to be fixed.

One window-end checkpoint scalar matches Torch at step 16
(0.98014408…). That scalar does not cover every head, every pool, or
the rest of the window. Torch typed views of `fill` / `p` / `w` stay
at the step-0 values (fill=1, p=0.99972, w last element 1) through
step 15, then agree at step 16. The Torch reference advances every
step: fill goes 1, 2, 3, …, 16, then back to 1, and p leaves 0.99972
on step 1. The same run reports that AscendC flush due-scan and fold
fired. Those two observations disagree. Open causes include a missing
stream sync, a stale snapshot or typed view, a different storage,
offset, or executor address, an aliasing or out-of-bounds write, and a
compiler or runtime fault. The record does not pick one. Calling it a
bookkeeping-visibility bug would skip that list.

Production `gdn.py` does not read these views. That limits how the
current end-to-end path can observe the disagreement. It does not make
the disagreement diagnostic-only, because a host due-gate or a later
recycle that trusts the view would follow the frozen values, and an
address mismatch would still be a state bug. `.long()` on a tensor that
is already int64 does not allocate; a fresh address is `data_ptr()`
after an explicit clone, not the dtype cast by itself.

After the v2 custom op, ordinary Torch advanced indexing hits IndexCheck
/ aclnnMul 507035. That is a separate blocker. The two-phase test keeps
the v2 warmup out of Phase A so the reference can run. It does not make
that runtime combination an acceptable production constraint. A slot
index that changes every step also flakes in pure Torch, so this test
kept slots stable and recycled windows with `init_from_dense`.

`VLLM_ASCEND_GDN_LQ_STEP_V2` stays off. The E1 gate 389≥279 is still
open. The reproduction note for an upstream issue is a draft on the
container. It has not been filed. The next localization, before any
wiring, is: write a distinct sentinel into each pool and print storage
pointer, offset, stride, and the address passed to aclnn; after an
explicit sync, copy the whole backing buffer to CPU and decode
`fill`/`p`/`w` from byte offsets; read the same address with a small
device kernel; compare device readback, CPU bytes, and the Torch view;
then repeat A/B/A with the executor cache off and on.

## What the measured slowdown does not decide

Every completed load on this port and on the vendored fork is slower
than dense: 0.78–0.80× on the fork's independent 8K scan, 0.77× on the
0.27.1 eager B=2 run, 0.54× on the 96-way ramp. That shows this NPU
implementation is slower. It does not show that LeapQuant has no gain
on a 910C. The slow path is the one the paper speeds up, and this tree
still computes it by rebuilding the state. The author baseline is
NVlabs/kda PR #10 at `48f02e6`. Its B200 times are a structure
reference. They are not a 910C target.

`gdn_leapquant_step.py` builds a full `S[128,128]`, adds the rank-4
compensator as outer products and up to 16 window outer products, then
computes `S·k` and `S·q`. It does not store `S` back to HBM, so the
traffic cut is real. A rank-1 term written as `S += u kᵀ` then `y = Sx`
touches `V×K` elements. The same term as `y += u (kᵀ x)` is a dot and a
vector update, `O(V+K)`. The int8 checkpoint matvec remains either way.
Register pressure and spill are not in a profile, so the source does not
prove the rebuilt `S` spills. The earlier direct-readout listing omitted
the compensator loads, divided `s_v` by 127 a second time, and did not
match this formula. Its B=8 timing does not test this direction. The
author step baseline at batch 256 is 0.0449 ms per layer, L2-cold, on
B200.

Production flush v1 is `KERNEL_TYPE_AIV_ONLY`. The rebuild, subspace
iteration, orthogonalization, and residual run on the vector core, with
row loops, `GetValue`/`SetValue`, and `SyncV2S()`. The author flush at
the same batch, also on B200, is 0.2465 ms all-due, 0.005 ms
nothing-due, and 0.031 ms at about 1/16 due. A synchronized window that
folds once every 16 steps would amortize as
`0.0449 + 0.2465/16 ≈ 0.0603` ms. Async requests need a mixed-due
measurement. That formula does not replace it. On the NPU the open
split is still nothing-due launch, scan, bind, and submit versus the
real fold. The fold does not have to stay a vector-only kernel. A
replacement can take pieces from the v2 chain. The decision is the
full-window cost against same-machine dense.

The B=2 eager wall cannot host the paper's decode claim. Dense GDN on
that run is `1.90/41.9 ≈ 4.5%` of the wall. If every other cost stayed
fixed, deleting that 1.90 s would leave `41.9/40.0 = 1.0475×`. Extra
reconstruct, quantize, or launch cost turns that into a loss. Matching
dense there is an integration check. The paper's pure-decode comparison
uses CUDA graphs. At batch 512 and 4K context on one B200 it reports
1.22–1.37× decode throughput (Figure 8). The separate "up to 1.4×"
figure is concurrent requests for Qwen3.5-9B on one B200, with prefix
caching and about one checkpoint per 1K cached tokens. Kernel speedup,
decode throughput, and admission capacity are three numbers. The 96-way
ramp is also slower, so the gap is wider than small batch alone. The
fork rows at N=256–512 count requests, not the batch that ran on each
decode step. A comparable run records that batch and the pure-decode
time.

The paper keeps a high-precision window too. It allocates that window
once per active request, about 8 KiB per head, and the persistent prefix
checkpoint is only the compressed boundary. This slot packs all three
into one head block: checkpoint 19,456 B, window plus metadata 8,264 B
(`_LAYOUT_END − _OFF_KBUF`), allocation 32,768 B. Across 32 heads the
checkpoint payload is 622,592 B and the allocated slot is 1,048,576 B.
The page line is `4,243,456/1024 = 4144` B/token in mode 0 and
`2,146,304/512 = 4192` B/token in mode 3 pad, so that line does not get
cheaper when the page and the block shrink together. That identity does
not account for the KVCacheManager +26%. Attention bytes, group layout,
padding, and the free-byte budget are still unlisted. Matching the paper
means a fixed checkpoint interval, a compressed persistent state, an
active window allocated on its own, and an allocator that sees the
smaller bytes. A page-size switch does not do that. The compact page
stays off.

A GPU that stays busy is not a throughput curve. A larger batch can
raise weight reuse, or it can already be saturated. The 0.54× ramp ratio
already includes the batch, the scheduler, prefill, and the generated
length that the run actually had. It is not multiplied by 1.145 or by
1.26. For a steady pure decode the useful ratio is

```text
R = (B_lq / T_lq(B_lq)) / (B_dense / T_dense(B_dense))
```

If the runnable batch grows by `c`, break-even is
`T_lq(cB) ≤ c · T_dense(B)`. That curve is not measured. Comparing the
B=2 counterfactual 0.876× with `1/1.145 ≈ 0.873` or with `1/1.26 ≈ 0.794`
mixes two regimes, and the inequality runs the other way inside that
invalid product: 0.876 is larger than both. The evidence that stands is
narrower. Every measured load is slower. The startup budget has not
become a serving gain. Whether a larger batch compensates is unmeasured.
There is no positive serving result yet.

The speed gap and the numerical contract are separate. Torch
`_quantize` rounds `u` and `q` to fp16 and then forms the residual from
those rounded values. Flush v1 forms the residual from fp32 `uk` and
fp32 `qv`, and casts both to fp16 only at the store (`CAST_RINT`). The
two checkpoints differ. That is not, by itself, a failing kernel. The
parity already recorded is an aggregate error. The author flush criterion
is per head, on the decoded state: at most 1.01× the reference error
when that error is at least `1e-4` of mean `|S|`, otherwise at most
`1e-3` of mean `|S|`. Their published table lets several fp16
intermediate choices pass that flush bound. Rounding the decoded state
to fp16 fails their step bound (1.66× the tolerance). Rounding the state
to bf16 fails both. This tree has not run that criterion, including
rank-deficient, tiny, and hot-row heads. A short token-exact trace does
not establish near-FP32 task scores. A later divergence does not, by
itself, fail them either.

Three gates, in order:

| gate | question | what it unlocks |
|---|---|---|
| operator | A formula-correct direct readout plus flush, amortized over a full window, faster than same-machine dense from a small batch through 256/512 | production performance wiring |
| engine | At the same actual batch, preprocess, binding, and graph execution still leave that operator gain | a real capacity boundary |
| serving | Under a fixed memory budget and the accuracy bar, compressed checkpoints raise useful serving throughput | an end-to-end claim |

The allocator rewrite waits on the operator gate. If a direct readout
and a fold that actually run on the matrix unit still lose to dense
after that measurement, that result is the 910C bound. The eager-torch
window below is not that bound. Production flags stay off. The old
direct-readout kernel stays out of the tree.

## E0a, E2, and the eager operator gate (910C, 2026-10-09)

Chip Ascend910_9382, 64GB, CANN 9.1.0, vLLM 0.27.1 + vllm-ascend
0.27.1rc1 at `3b3188623`. The container tree has 20 dirty files on top
of that SHA, so the SHA alone does not name the measured build.
`libcust_opapi.so` SHA256 starts `5cfdb993`. E0a passed 12/12: the six
LeapQuant modules resolve in the workspace tree, the flush probe
returns without a slow-path, and the harness bindings do not import
`fla_npu`. New kernel timings can be cited from this run. The scripts
and `results/repro_0271/{e0a,e2,e3,e5}/` stay on that host. They are
not in this tree. E0b and the step-v2 address matrix were not run.

E2 called flush v1 with `ITERS=50` on one chip. `B` is the length of
`idx` in the call. Correctness passed: slots in the call that were not
due kept their bytes; due slots folded (`w=0`, `p=1`, `fill` left at
`L`, codes changed, outputs finite).

| B | due = 0 | due ≈ 1/16 | due = all | all-due / B |
|---:|---:|---:|---:|---:|
| 2 | 0.017 ms | 0.398 ms | 0.400 ms | 200 µs |
| 64 | 0.012 ms | 8.411 ms | 8.416 ms | 131 µs |
| 256 | 0.022 ms | 33.43 ms | 33.45 ms | 131 µs |
| 512 | 0.033 ms | 66.85 ms | 66.87 ms | 131 µs |

Nothing-due returns in 0.012–0.033 ms. On this due matrix a count of
about 1/16 costs the same as every slot being due, to 0.03 ms. For
`B≥64` the all-due column is 131 µs per slot in the call, 4.1 µs if
that column is also divided by 32 heads. `B=2` is 200 µs per slot and
is not on that line. Those quotients describe this matrix. A later
placement of one due sequence against four moves the same `B=64` wall
from 6.26 ms to 8.21 ms, so the 1/16 row matched all-due because that
pattern filled the slowest block. A fresh `idx` allocation on the
nothing-due path costs 0.222 ms against 0.017 ms at `B=2`; at `B≥256`
the report says the pointer stays put and the penalty drops. The
B=2 engine average of 0.344 ms per layer-step sits between the
nothing-due and any-due rows, which is what a mix of those calls looks
like. It is not a constant to reuse at a larger batch.

A follow-up on the same chip held `B` fixed and raised the allocated
pool `S` from 16 to 512 (`ITERS=20`, logs `e2/B2_S512.log` and
`e2/B64_S512.log`). The cost did not change. The bill is the length of
`idx`, not the allocated pool. The pasted summary gives that comparison
and not the raw milliseconds. One line,
`0.14 ms + 131 µs × B`, is 0.402 ms at `B=2` against the measured
0.400 ms. The same line is 8.52 ms against 8.416, 33.68 against 33.45,
and 67.21 against 66.87, high by about 0.5–1.3%. The 200 µs per slot at
`B=2` is that 0.14 ms fixed term spread over two slots, on top of the
131 µs slope.

`gdn.py` calls `_lq_flush` on every decode step, and `idx` is the
current decode batch, not the whole KV allocation. The `S` sweep is
the measurement behind that sentence. E5's window flushes
once per 16 lockstep steps, so it pays one due call per window. Reading
this table as "any due pattern bills the all-due wall" made a staggered
batch look like 33.4 ms of flush on every step at `B=256`, about 534 ms
across 16 steps. The bisect below retires that reading: the wall is the
due count on the slowest block. The 534 ms product is not a serving
number.

E3 checked the direct algebra on a frozen checkpoint, both Torch
`_quantize` and a real flush-v1 fold. Single-term activation, a 96-step
same-state replay, and the FP64 formula oracle are at most 1e-5. BF16
outputs against the author step gate are relative L2 1.47e-3 and
1.28e-3, and max-abs over the head max 5.85e-3 and 5.03e-3. Against a
dense FP32 oracle the aggregate errors are 4.88e-3 and 1.35e-2, inside
the old 2e-2 envelope and outside the author per-head gate, which does
not apply to an independent dense trajectory. Writeback of codes,
scales, `w`, `p`, `fill`, and `kbuf` matched on an independent 32-step
replay. `ubuf` is the bf16 archive of the delta update and stayed
inside 3e-3 of the head max. `decode_state` reconstructs the checkpoint
only. A live window has to go through `reconstruct_dense`. An F1
checkpoint is a flush that actually ran at `fill == L`. Same-state
application is the algebra comparison; a second pool that also flushes
mixes in a new quantization.

Eager Torch direct versus materializing, with writeback, fill fixed:

| B | fill 0 | fill 8 | fill 15 |
|---:|---:|---:|---:|
| 2 | 0.78× | 0.69× | 0.72× |
| 16 | 0.69× | 0.69× | 0.69× |
| 64 | 1.07× | 1.07× | 1.07× |
| 128 | 1.24× | 1.24× | 1.23× |
| 256 | 1.34× | 1.35× | 1.34× |

At `B=256` the materializing step is 11.75 ms and the direct step is
8.74 ms. The 10% bar against the current step is met at 128 and 256,
and fill does not move it. At `B=8` an earlier pair was 1.644 ms
against 2.204 ms, direct slower. The crossover is between 16 and 64.
This ratio is against the materializing eager step. It is not `R_kernel`.

E5 is the operator gate on the same chip, eager Torch on both sides,
three paired rounds, six windows, fill left in place. `D` is the dense
FP32 recurrence in that eager framework. It is not the production fused
dense kernel. `M` is the materializing step plus one flush-v1 call per
16 lockstep steps.

| B | D ms/window | M ms/window | flush in M | R = D/M |
|---:|---:|---:|---:|---:|
| 16 | 4.20 | 31.99 | 2.17 | 0.131 |
| 64 | 14.94 | 64.01 | 8.42 | 0.234 |
| 128 | 29.76 | 104.59 | 16.83 | 0.285 |
| 256 | 59.15 | 188.23 | 33.44 | 0.314 |

The paired range at each `B` is at most 0.111–0.138, and at 256 it is
0.314–0.316. The 1.10 bar fails. Flush is 6.8% of the `B=16` window
and 13–18% from `B=64` up. Replacing the step with the eager direct
time and keeping the measured flush gives, at `B=256`,
`8.74×16 + 33.44 = 173` ms and `R ≈ 0.34`. The 0.42 figure is the
step-only ratio ` (59.15/16) / 8.74 `. It is not the window ratio.
Either way the eager direct path loses. Eager Torch materializes the
decoded state, so the author's traffic cut, about 28 KiB read per head
against a 64 KiB fp32 state that is also written back, is not present
in these timings. Codes alone are 16 KiB. A fused kernel that reads
codes, scales, and the window once and does not build `S` is what can
expose that cut. This fail does not measure that kernel.

Stop here for the engine and the allocator. E6 and E7b stay unstarted.
The fused direct step is reported on the host. The fold bisect below
puts the pre-change flush at about 195 µs per due instance on the
slowest block. A later host pass cut that to about 142 µs and paired
the window against the production dense op. That pairing fails, and it
is recorded after the bisect. Production flags stay off.

## Fused Triton direct step, and the any-due floor (910C, 2026-10-09)

`ops/triton/gdn_leapquant_step_direct.py` and
`tests/test_step_triton_direct.py` are on the container. They are not
in this tree. The kernel reads codes in one pass for the two matvecs
and applies the window as `O(L(V+K))`. It does not build `S`.

The reported 96-step author-gate errors are relative L2 1.47e-3 and
max-abs over the head max 5.85e-3, with writeback matching the
materializing step. Those two errors are the same digits as the earlier
eager-direct F0 row. The Triton log is the source for this kernel; the
match alone is not a second measurement. Direct-Triton step time is
0.170 / 0.575 / 1.197 / 2.411 ms at B=16 / 64 / 128 / 256. The report
puts that at 1.23–1.39× the materializing Triton step. Against the E5
eager dense window, one step at B=256 is `(59.15/16) / 2.411 = 1.53×`
(the report says 1.55×). That is a step ratio. The eager dense kernel
is still not the production fused dense kernel. A later run on the
same host did not reproduce 0.170–2.411 ms. The step-kill section
uses the times from that run.

A lockstep window estimate, not a rerun of E5, is `16 × step +` the
flush already measured inside the E5 `M` window:

| B | 16 × direct step | flush | estimated window | eager D | R |
|---:|---:|---:|---:|---:|---:|
| 16 | 2.720 | 2.17 | 4.890 | 4.20 | 0.859 |
| 64 | 9.200 | 8.42 | 17.620 | 14.94 | 0.848 |
| 128 | 19.152 | 16.83 | 35.982 | 29.76 | 0.827 |
| 256 | 38.576 | 33.44 | 72.016 | 59.15 | 0.821 |

`R` is 0.82–0.86. The 1.10 bar is not met, and this table was not
measured as a paired window. Flush is 44% of the B=16 estimate, 48% at
B=64, and 46% at B=128 and B=256. The direct step is the other half.
Dropping the flush term on paper would put B=256 at `59.15/38.576 =
1.53`, which is why the flush call is now the piece that keeps the
estimate under 1. A formal E5 rerun is still required before that
sentence becomes a result. E6 and E7b stay stopped.

The same host then split one due head from four due heads at B=64
(`ITERS` not restated; due0 matches the earlier nothing-due row):

| pattern | time |
|---|---:|
| due 0 | 0.011 ms |
| one due head | 6.258 ms |
| four due heads, spread | 6.263 ms |
| four due heads, at the tail | 8.212 ms |
| B=2, all due | 0.400 ms |

B=64 all-due remains 8.416 ms. One due head is already 6.258 ms, about
48× a 131 µs per-slot line. Four spread heads cost the same as one
head. Four heads at the tail cost 8.212 ms, so placement still moves
about 2 ms. Dividing those walls by `B × 32` gave 3.1–3.8 µs per
instance in the call. That denominator counts idle blocks. The bisect
below replaces it.

`trace_head1.json` sums the `GdnLqFlush` device kernel to 31.303 ms
over 5 launches, 6.261 ms each. That matches the 6.258 ms wall. The
6.26 ms is kernel time. The trace's Computing bucket is 33.36 ms, and
its Free time is 245 ms of host idle in the capture, not a per-call
submit cost. Nothing-due at B=64 is 0.011 ms. Divided by every instance
in that one call it is 5.4 ns. The B=2 nothing-due call is 0.017 ms,
0.27 µs per instance, so the 5 ns figure is not a nothing-due law.

## Flush fold bisect (910C, 2026-10-09)

`msprof` on `bench_flush_due_matrix`, `B=64`, 20 iterations,
PipeUtilization, `collect launch`. The kernel is `AI_VECTOR_CORE`, 48
blocks. The counters that move are the `aiv_*` columns. The `aic_*`
columns stay 0. Nothing-due duration is 12.7 µs, busy 10.4 µs (mte2
3.8, scalar 2.9, vector about 0). Calls with dues take 8.41–8.42 ms.
The four-due-batch call is busy for 531 µs, 94% idle: that idle is the
average across the 48 blocks, one hot block against idle blocks. The
all-due call is busy for 8348 µs, 99% busy, about 194 µs of busy time
per fold on a full block. MTE2 on the slow path is 14–25 µs. The
4-byte `DataCopyPad` L2-miss account of this wall is false. The time
is in the vector and scalar pipes.

The same host added a third attribute, `dbg_mode`, default 0, a tiling
field, and the env `GDN_LQ_FLUSH_DBG`. The executor cache key includes
the mode. `ProcessDue` returns at the phase named by the mode.
`scripts/bench_flush_dbg.py` times the modes and does not recheck
correctness. The attribute, the script, the sweep log, the `msprof`
dump, and the kernel backup stay on that host. They are not in this
tree. After that build, E2 bench and the AscendC flush parity check
passed again with `dbg_mode` left at 0. Keeping the attribute upstream
is undecided.

Three build facts, each hit once on that host. `build_aclnn.sh` rebuilds
the host side. The device kernel that runs is the configure-time copy
under `csrc/build/binary/ascend910_93/src/<op>/op_kernel/`, and a
`gen/<op>_ascend910_93_0.done` marker skips it. The first sweep, every
mode at 8.41 ms, was that stale binary and is void. Forcing the rebuild
is deleting the marker and ninja-building the
`GdnLqFlush-gdn_lq_flush-0.sh` rule with the CANN environment. Adding
the attribute without regenerating `binary/.../src/gdn_lq_flush/GdnLqFlush.py`
made opc report E80003: the new value landed in `kernel_name`. The host
patched `_build_args`, the `check_op_params` decorator, and the three
signatures in that generated driver. A later edit of a `gdn_lq` kernel
refreshes the `binary/src` copy and the generated driver and deletes
the marker, or wipes `binary/` and reconfigures.

Bisect at `B=64`, event median, milliseconds. `head1` is one due
sequence: 32 head instances, all on block 0. `head4` is four due
sequences, 43 instances on the slowest block. `spread4` keeps every
block at or under 32. `dueall` is 43 on every block. Mode 10 returns
at the `ProcessDue` entry. Mode 0 and mode 9 are the full kernel.

| mode | stops after | head1 | head4 | spread4 | dueall |
|---:|---|---:|---:|---:|---:|
| 10 | entry, no fold | 0.0096 | 0.0092 | 0.0095 | 0.010 |
| 1 | `p` read, MTE2 and a barrier | 0.0146 | 0.0189 | — | — |
| 2 | nine GM reads, 29 KB | 0.0224 | 0.0296 | — | — |
| 3 | casts, the `w` loop, `SyncV2S` | 0.036 | 0.047 | — | — |
| 4 | pass 0+1, S rebuild and qk | 3.193 | 4.289 | — | — |
| 5 | `Mgs4` of qk | 3.256 | 4.373 | — | — |
| 6 | pass 2 and `Mgs4` of qv | 4.430 | 5.951 | — | — |
| 7 | pass 3 and pass 4a | 5.480 | 7.362 | — | — |
| 8 | pass 4b quantize, no writeback | 6.243 | 8.387 | — | — |
| 0/9 | full, with writeback | 6.253 | 8.400 | 6.256 | 8.412 |

Mode 10 on `dueall` is 0.010 ms. The stale-binary 8.41 ms at that mode
is void. `head1` matches `spread4` because both put about 32 dues on
the slowest block. `8.400 / 6.253 = 1.343` and `43 / 32 = 1.344`.
`6.253 / 32 = 195.4` µs and `8.412 / 43 = 195.6` µs. `43 × 195` µs is
8.39 ms against the measured 8.412. At `B=2`, all-due leaves 2
instances on the slowest block, and `2 × 195` µs is 0.39 ms against
the measured 0.400. At `B=256`, all-due is 8192 instances, 171 on the
slowest block, and `171 × 195` µs is 33.3 ms against the measured
33.45. The earlier one-due-sequence point at `B=16`, 1.962 ms, is 11
instances at 178 µs if that block held 11, about 9% under 195 µs. The
195 µs line is the `B=64` and `B=256` all-due rate. It is a close
description of `B=16`, not an exact one.

Head1 phase gaps divided by 32, and the same gaps on head4 divided by
43 for pass 0+1:

| phase | head1 | share of 195 µs |
|---|---:|---:|
| pass 0+1, S rebuild | 98.7 µs | 51% |
| pass 2 and `Mgs4`(qv) | 36.7 µs | 19% |
| pass 3 and pass 4a | 32.8 µs | 17% |
| pass 4b quantize | 23.8 µs | 12% |
| `Mgs4`(qk), the reads, casts, writeback | 3.4 µs | 2% |

Pass 0+1 on head4 is `(4.289 − 0.047) / 43 = 98.7` µs, the same rate.
The paste rounded the last row to about 2.5 µs. The head1 remainder
after the four large phases is 3.4 µs. The PipeUtilization reading on
the slow path is about 10.8K vector instructions of 128 elements and
159 µs of vector-pipe busy time, about one thirteenth of that pipe's
peak. That is the counter. Pass 0+1 in
`gdn_lq_flush.cpp` is the `v` loop that loads `sv`, `q`, `ubuf`, and
`q0` with scalar `GetValue` and then issues `Muls` and `Add` for that
row. The kernel already marks those `GetValue`s as S-pipe reads that
need `SyncV2S`. The phase split plus the MTE2 counter put the wall on
that fold. On the all-due call the vector pipe is busy for 159 µs and
the same reading puts it at about one thirteenth of its peak, which is
the room a scalar `GetValue` in front of each row's `Muls` would leave.
An instruction trace that names the stalled opcode is not in this run.

The scan is the 0.010 ms mode-10 call. `Process` loads 4 bytes of
`fill` per instance (`DataCopyPad` of 4) and one `PIPE_ALL` per
`SCAN=32` group, then calls `ProcessDue` only when `fill >= L`. Those
loads also run when nothing is due. Successive heads are `HEAD_BYTES`
apart, 32768 bytes, so one 128-byte contiguous read of a batch row
does not cover 32 heads. The wider fill read, and a split that only
reorders scan and fold, are retired. Scan and launch are already
microseconds.

A fold at 20–40 µs is a target. Nothing at that rate has been timed.
Scaling the measured `B=256` lockstep estimate (direct-Triton step
`16 × 2.411 = 38.576` ms, flush 33.44 ms, eager `D` 59.15 ms,
`R = 0.821`) by `t / 195` on the flush term:

| fold | flush | window | R vs eager D |
|---:|---:|---:|---:|
| 195 µs, measured | 33.44 | 72.02 | 0.82 |
| 96 µs, pass 0+1 removed and the rest kept | 16.5 | 55.1 | 1.07 |
| 89 µs | 15.3 | 53.8 | 1.10 |
| 40 µs | 6.86 | 45.4 | 1.30 |
| 20 µs | 3.43 | 42.0 | 1.41 |

Clearing 1.10 against this eager `D` takes the whole fold to about
89 µs or under. Removing only pass 0+1 leaves about 96 µs and
`R ≈ 1.07`, which misses 1.10. This table is the eager-`D` scale.
The production dense op, measured in the next section, takes 15.98 ms
for the B=256 window, and the retimed flush on that pairing is 24.33 ms.
The 1.10 column does not carry over to that op.

One sequence is 32 contiguous instances (`inst = b * H + h`, `H = 32`).
At `B=256` a block holds about 171 instances, so those 32 heads sit on
one block unless the sequence straddles a boundary. A staggered step
whose slowest block holds one such sequence is `32 × 195` µs, 6.2 ms,
and several due sequences on that block cost more. Striping the 32
heads across blocks would be a different schedule, about 11 dues per
block and 2.1 ms, and this kernel does not stripe them. Neither
product is an engine trace, and neither is a serving number. The
retired 534 ms figure is the one that billed the all-due wall on every
step.

## De-scalarized flush, and the production dense window (910C, 2026-10-09)

The same host edited `GdnLqFlush` and kept the edits that passed the
parity gate. The kernel snapshot, `scripts/bench_flush_dbg.py`, and
`scripts/bench_window_vs_dense_op.py` stay on that host. They are not
in this tree. `dbg_mode` stayed at 0. The host's recommendation is to
land the attribute and both scripts with the kernel patch. E2 bench
passed four times, including its correctness checks. The AscendC flush
parity check passed with reported max 0.01021, the same gate as the
pre-change run. The parity suite was 6/7. `test_step_v2_multipool`
failed on the w/p/fill visibility assertion. The host says the 10:51
log has the same signature, which is the executor bookkeeping issue
already recorded above, so this round does not add a regression. An
e2e smoke of mode 3 was 4 prompts in 3.9 s against a 4.0 s baseline.
At B=2 the flush is a small share of that smoke. It is not an e2e win.

The adopted fold edits, as the host describes them:

1. Pass 0+1 uses `Axpy` for the window and compensator terms, and for
   the qk accumulation. The host's phase timer went from 99.8 µs to
   65.3 µs per instance. `Axpy` is `dst += src × scalar`. This build
   left that op enabled. The host puts the fused rounding within 1 ulp
   of `Muls` plus `Add`, inside the relative-diff gate.
2. Pass 0+1 overlaps two rows: the next row's scalar reads sit under
   the current row's vector issues. 65.3 µs went to 59.2 µs per
   instance, a 9% cut.
3. Pass 2 gives each dot its own staging row. The host reports that
   this removed a write-after-read on the shared halving buffer. The
   final pass-2 rate below did not move.
4. Pass 3 accumulates with `Axpy(uk_r, srow, qv[r,v])`. Pass 4a applies
   `E = S − Σ c·uk` as an in-place `Axpy` with a negative scalar. The
   host's timer for pass 3+4a went from 32.8 µs to 20.9 µs, a 36% cut.
5. Pass 2 and pass 4b issue one V-to-S sync per 8 rows instead of per
   row, 64 syncs down to 32 per instance. The final pass-2 and pass-4b
   rates below did not move.

Final phase profile at B=64, `head1`, 32 instances on the slowest
block. The full call is 4.551 ms, which is 142.2 µs per instance.
The host's per-phase rates, in µs per instance:

| phase | µs |
|---|---:|
| scan | 0.32 |
| `p` read | 0.15 |
| GM reads | 0.65 |
| pass 0+1 | 59.3 |
| `Mgs4`(qk) | 1.9 |
| pass 2 and `Mgs4`(qv) | 36.6 |
| pass 3 and pass 4a | 21.0 |
| pass 4b | 23.2 |
| writeback | 0.4 |

Those nine rows sum to 143.5 µs. The full-wall rate is 142.2 µs. The
pre-change head1 wall was 6.253 ms, and `6.253 / 4.551 = 1.374`.
Against the pre-change phase split, pass 0+1 moved from 98.7 µs to
59.3 µs and pass 3+4a from 32.8 µs to 21.0 µs. Pass 2 stayed at
36.6 µs (was 36.7) and pass 4b at 23.2 µs (was 23.8). The 99.8 µs
starting point in the iter list is the host's phase timer, 1 µs above
the 98.7 µs recorded from the earlier head1 gap.

E2 on the new binary. Nothing-due stays in the 0.01–0.03 ms band:
0.011, 0.021, and 0.032 ms at B=64, 256, and 512. Newly measured due
times, with the recorded all-due time beside them:

| B | due ≈ 1/16 | all-due | recorded all-due |
|---:|---:|---:|---:|
| 2 | 0.292 | 0.298 | 0.400 |
| 64 | 6.120 | 6.127 | 8.416 |
| 256 | 24.32 | 24.33 | 33.45 |
| 512 | 48.64 | 48.65 | 66.87 |

On this matrix the two due columns still match each other. The paste's
before-value at B=2 all-due is 0.403 ms against the recorded 0.400, and
at B=256 it is 33.68 ms, the old fit `0.14 ms + 131 µs × B`, against
the measured 33.45. The B=512 before-value in the paste is 66.87 ms on
the one-sixteenth column. Ratios against the recorded all-due column
are `0.400 / 0.298 = 1.34` at B=2 and 1.375 at B=64, 256, and 512.
`6.127 ms / 43` instances is 142.5 µs. At B=256 the slowest block holds
171 instances, and `171 × 142.4` µs is 24.35 ms against the measured
24.33. The 142 µs line replaces the 195 µs line for this binary. One
sequence still occupies one block, so the same untraced staggered
placement is `32 × 142` µs, 4.55 ms on the slowest block. That is still
not a serving number.

The host's msprof difference across modes 4, 5, 6, and 8, taken before
the edit, reports about 98% idle on each phase, with 87 µs of vector
busy and 51 µs of scalar busy against a 5.15 ms wall. A launch whose
only hot block is one of 48 is already about 98% idle as a chip
average. The earlier all-due capture was 99% busy when every block
worked. The paste does not say whether 87 µs and 51 µs are the hot
core or that average. The wall-clock experiments are the evidence for
the dependency chain. Three pass-2 shapes (shared tmp plus a reduce,
private staging plus a reduce, a halving tree) cost the same wall, and
the final pass 2 is still 36.6 µs. Overlapping two rows hid part of
the scalar delay and won 9%. Replacing the `Muls`/`Add` tmp chain with
`Axpy` won about 35% on pass 0+1 and 36% on pass 3+4a. Further ISA
work is unscheduled: a reduce-port GEMM the host bounds near 50 µs per
instance, a MIX/Cube pair of small GEMMs, or an 8-wide broadcast
multiply across the plane.

The production dense window was paired on this host. `D` is
`torch.ops._C_ascend.npu_recurrent_gated_delta_rule`, which the host
calls the mode0 path, registered through `enable_custom_op()`. This
checkout's `gdn.py` imports `recurrent_gated_delta_rule` from
`fla_npu.ops.ascendc`. The two names were not checked against each
other here. The op's state argument is a plain contiguous fp32 tensor.
The host says a pools buffer must not be passed in, because that
layout interleaves codes and `kbuf`. The `D` arm uses its own dense
tensor and starts from `Sd0`. Query and key are `[T, Hk, K]` bf16
after l2norm, `g` is `[T, Hv]` fp32 log-decay, `beta` is `[T, Hv]`
bf16, `actual_seq_lengths` is `[0, len, …]`, and
`num_accepted_tokens` has length `B`. Four steps of this op against
the eager dense state differ by relative 1.47e-3. Those digits match
the earlier author-gate rows, so this check does not reconfirm those
rows. `M` is still the eager materializing step plus one flush per 16
lockstep steps, now on the de-scalarized flush.

| B | D_op ms/window | M ms/window | flush in M | R = D_op/M |
|---:|---:|---:|---:|---:|
| 16 | 1.247 | 31.31 | 1.58 | 0.040 |
| 64 | 3.537 | 63.14 | 6.13 | 0.056 |
| 128 | 8.176 | 103.0 | 12.24 | 0.079 |
| 256 | 15.98 | 187.0 | 24.33 | 0.085 |

The 1.10 bar fails. `1.247 / 31.31` rounds to 0.040. At B=256 the flush is
`24.33 / 187.0 = 13%` of `M`, and the other 162.7 ms is the eager
materializing step, 87% of that window. The flush column is the new
E2 all-due time and is 1.375× under the flush column of the eager E5
table (`33.44 / 24.33`). The `M` window only moved from 188.23 ms to
187.0 ms, so the non-flush remainder rose from 154.8 ms to 162.7 ms
across the two runs. The host quotes an eager-`D` ratio of 0.315 at
B=256 on this run. The paired eager E5 already on file is 0.314 at
B=256. The 0.285 figure is the old B=128 eager ratio, not the
before-value of this B=256 row.

`D_op` at B=256 is 15.98 ms against the earlier eager `D` of 59.15 ms,
which is 3.70×. Per step that is 1.00 ms against 3.70 ms. The
lockstep estimate that replaces the eager step with the direct-Triton
step and keeps this flush is `16 × 2.411 + 24.33 = 62.91` ms, and
`15.98 / 62.91 = 0.254`. That estimate was not run as a paired window,
and its 2.411 ms step was not reproduced later. Clearing 1.10 against
this `D_op` would require the whole LeapQuant window to come in under
`15.98 / 1.10 = 14.5` ms. The step-kill section is the measurement
against that line. E6 and E7b stay stopped.

Two more build facts from this pass. `ninja` piped to a grep for
`error` is not a compile check: the `OPS_LOG` text matches, and
`build_aclnn.sh` still packages the old object. The mtime of
`binary/…/bin/gdn_lq_flush/*.o` is the check that the device binary
changed. Three timings that showed no movement were stale objects.
Vector operands have to be 32-byte aligned. A halving tree can stop
at 8 lanes, which is 32 bytes. Four, two, or one lane faults on
device with 507035, "UB address not aligned". That is why
`ReduceMaxToEight` stops at 8 lanes. The tail from 8 to 1 stays a
scalar `GetValue` or a reduce. This 507035 is the alignment fault.
It is not the step-v2 IndexCheck reported above.

## Step-kernel kill line (910C, 2026-10-09)

The line agreed from the production window: at B=256, R≥1.10 needs the
lockstep LeapQuant window at or under 14.5 ms. A flush at the
unmeasured 50 µs/instance bound is about 8.6 ms (`171 × 50` µs), which
leaves about 0.37 ms/step. The host used 0.4 ms/step as that kill
line. A direct step with no path to it stops the operator work: no
multi-head step, no further flush polish, no E5.

The same host retimed the direct step. Back-to-back amortization
matched the per-call figure, so the times are not a launch gap. The
§15 row of 0.170 / 0.575 / 1.197 / 2.411 ms was not reproduced.

| B | Triton direct | AscendC direct | 16 × AscendC | D_op window |
|---:|---:|---:|---:|---:|
| 16 | 2.573 | 0.299 | 4.78 | 1.247 |
| 64 | 3.788 | 1.074 | 17.18 | 3.537 |
| 128 | 5.278 | 2.134 | 34.14 | 8.176 |
| 256 | 8.728 | 4.253 | 68.05 | 15.98 |

The AscendC column is the existing direct-algebra step. At B=256,
`4.253 / 171 = 24.9` µs on the slowest block, and `4.253 / 0.4 = 10.6`.
The 16-step AscendC total divided by the production window is 3.8× at
B=16, 4.9× at B=64, 4.2× at B=128, and 4.3× at B=256.
The Triton column is slower at every B. No correctness figure is
attached to either column in this run.

The reduce probe reused the flush `dbg_mode` path. Modes 11, 12, and
13 time a `[128,128]` row sum as `wall(mode) − wall(4)`. Modes 21, 22,
and 23 read the same shapes back through the `s_k` slot and compare
them with a same-seed reference. The kernel, the driver, the binding,
and both bench scripts stay on that host. They are not in this tree.

| shape | µs per matvec | check |
|---|---:|---|
| 12/22, row `Mul`+`Add`, one-repeat `WholeReduceSum` (`CodesDots16`) | 7.6 | passes, fp32-level against the reconstruct reference |
| 11/21, pair-add, `WholeReduceSum(64, repeat=128)` | 1.0 | fails, max abs about 30 against mode 22 |
| 13/23, strided halving, `WholeReduceSum(8)` with a repeat | 1.4 | fails, max abs about 32 against mode 22 |

Both multi-repeat widths fail against mode 22, with max abs about
30 and 32. The host first read that as a broken multi-repeat
primitive. The production-header probe below reran the K=128 fold
with the production argument order and 64-row tiles. That form passed.
The 1.0 µs row in this table is a 128-row call with the stride
arguments in the wrong slots, and it is not a usable measurement of
the primitive. The host identifies the old failure with the step
kernel's earlier mis-scattered reduce. The shape that passes is the
one already in the step, at 7.6 µs. Two codes matvecs are 15.2 µs per
instance. On the B=256 slowest block that is `171 × 15.2` µs `= 2.60`
ms per step, and `16 × 2.60 = 41.6` ms, before the window term and
before the flush. Those matvecs alone are 2.9× the 14.5 ms budget. The measured 24.9 µs step is that 15.2 µs plus the
rest of the instance.

A correct multi-repeat at the 1.0 µs timing would be `2 × 1.0 × 171`
µs `= 0.34` ms of matvec per step. The paste's ~0.45 ms is higher
than that product. Sixteen steps of 0.34 ms are 5.5 ms, which would
leave 9.0 ms of the 14.5 ms budget for the rest of the step and for
the flush. The measured B=256 flush is 24.33 ms, so that path is
closed by the flush as well. It is closed first by the wrong results.

The remaining route the host names is an fp32 `TransDataTo5HD`
transpose plus a scalar `Axpy`, estimated at ~0.33 µs per matvec and
a ~0.45 ms step floor. `0.33` µs `× 2 × 171 = 0.11` ms, so the
0.45 ms floor is not that product. The host's stop reason is the one
that matters here: the parameters are undocumented, the fast
multi-repeat reduce has already shown that a quick primitive can time
well and compute wrong, and a success would still sit on the 0.4 ms
line with the window dots, the loads, and the conversion unaccounted
for. That route was not implemented.

After the probe was built into the flush kernel, flush parity passed,
E2 bench passed, and `dbg_mode` 0 was unchanged. Mode 0 read 4.551 ms
and 6.122 ms, the head1 wall and the B=64 all-due wall from the
de-scalarized binary (`6.127` ms there). The 6.12 ms figure is that
B=64 call, 43 instances at 142 µs. It is not the B=256 flush, which
stays 24.33 ms. E6 and the allocator stay stopped.

The 7.6 µs figure is the cost of the shape that passed. It is not a
hardware floor. The 0.4 ms/step line is the budget left after an
unmeasured 50 µs flush: `14.5 − 8.6 = 5.9` ms for 16 steps, about
0.37 ms/step, which the host rounded to 0.4. A free flush would leave
`14.5 / 16 = 0.91` ms/step. A flush that stays at 24.33 ms is already
over 14.5 ms, so no step time clears the bar. The current step
organization stays stopped. The official reduce below is the one
untested path that could change the 7.6 µs number.

`16 × 4.253 + 24.33 = 92.38` ms. `15.98 / 92.38 = 0.173`. That sum
stitches the AscendC step column to the paired flush. It is not a
paired window, and this run did not attach a correctness check to the
step column. The 0.085 ratio is the eager materializing window, not
this step.

## Official dense reduce, not yet probed (vllm-ascend `3b3188623`)

The production op's kernel entry is marked
`KERNEL_TYPE_AIV_ONLY`. Missing a cube unit does not by itself
explain a gap against that op. This tree's earlier cube GEMV probe
and the MIX launch failure are a different path. The `.cpp` selects
the kernel header with `__CCE_AICORE__`:

- `__CCE_AICORE__ == 310` builds `arch35/recurrent_gated_delta_rule.h`.
  For K=128 it adds the two vector registers of a row and reduces
  that register, one row at a time, then stores the first element.
- Any other core builds `recurrent_gated_delta_rule.h`. For K=128 and
  a tile of at most 255 rows it folds the two halves with one
  multi-row `Add` into `src[64]`, so the source and the destination
  of that `Add` do not overlap, barriers on `PIPE_V`, and issues one
  `WholeReduceSum` of 64 lanes with `repeatTime` equal to the row
  count, destination repeat stride 1, and source repeat stride
  `alignK / 8`.

The second header is the one quoted as the production fast path. The
first is what a 310 build actually runs. Which header the 15.98 ms
binary contains is not known. This checkout's `gdn.py` imports
`fla_npu.ops.ascendc.recurrent_gated_delta_rule`, while the timed op
was `torch.ops._C_ascend.npu_recurrent_gated_delta_rule`. The step
kernel in this tree, `CodesDots16`, already folds each K=128 row to
64 lanes and then issues `repeatTime=1`. Its comment records that a
repeated `WholeReduceSum` mis-scattered sums and that the
`dstRepStride` unit depends on the architecture. The failed probe
shapes are not in this tree, so the max-abs of about 30 is not yet
identified with that overlap, nor with a stride mismatch against a
packed reference.

The next measurement is one probe, on the header this chip compiles,
and it stops again if that reduce is wrong or is not faster than
7.6 µs per matvec. The loaded library's hash is the identity of the
binary. An object mtime only says a rebuild happened. A paired
16-step window is not part of this probe.

Zeroing the two codes matvecs does not enter the budget below. They
are 15.2 µs of the measured 24.9 µs instance, from the flush-debug
probe identified with `CodesDots16`, not from a split of the 4.253 ms
trace. `4.253 − 2.60 = 1.65` ms/step for everything else, and
`16 × 1.65 = 26.4` ms. That is already over the 14.5 ms window,
before flush and before whatever of the 15.2 µs remains. If the
in-step dots are cheaper than the probe, this remainder is larger.
Splicing a faster reduce into the current step leaves that remainder.

`15.98 / 1.10 = 14.53` ms is the whole lockstep window, step and
flush together:

| step | 16 × step | flush has to be under |
|---:|---:|---:|
| 0.30 ms | 4.80 | 9.73 ms |
| 0.40 ms | 6.40 | 8.13 ms |
| 0.50 ms | 8.00 | 6.53 ms |
| 0.80 ms | 12.80 | 1.73 ms |

The 0.80 ms row needs a flush under 1.73 ms, which is about 10 µs per
instance on the slowest block. Pass 4b alone is 23.2 µs, so that row
is not a target. The rows that still meet an 8.6 ms flush, the
unmeasured 50 µs bound, need the step at about 0.37 ms or under.
Today's 4.253 ms step and 24.33 ms flush are outside every row. Both
have to move. Fusing those two kernels into one launch keeps both
costs.

The codes term can fold `(s_k ⊙ x) / 127` into `x` before the matvec,
and apply `p` and `s_v` to the result. The compensator and the window
stay; they are not removed by sharing one codes tile across the key
and query readouts. GQA here is 16 key heads and 32 value heads. The
dense kernel indexes `head_i / (NV / NK)` inside the head loop.
Dropping the host `repeat_interleave` removes that expand. It is not
a 2× cut of the matvec. The first version that gets timed stays fp32.
The cube path and a hi/lo split wait. MIX still does not launch.

1. Record the loaded library hash and which of the two headers that
   build selected.
2. Compile that header's K=128 reduce for the same tile height, and
   compare it with the single-repeat reference on the layout the
   header consumes. The inputs cover random values, zeros, tiny
   values, a large row, and cancellation. A wrong result or a time
   that does not beat 7.6 µs ends the probe. Nothing else starts.
3. Only after that passes, a new direct step may load each codes tile
   once, decode it, run both readouts, and keep the compensator and
   the window. It has to come in near the 0.3–0.5 ms rows. A kernel
   whose non-matvec part is still ~1.65 ms does not.
4. A flush rewrite is the same header applied to the fold's
   reductions, not a merge of the current kernels. It has to come in
   under the flush column of whichever step row is real. Nothing-due
   is already 0.01–0.03 ms, so skipping the scan does not move these
   rows. One head stays on one block. Splitting that head's value
   tiles across blocks means the fold cannot run in the same kernel.

Rebalancing heads across blocks can change a mixed-due wall. An
all-due call already occupies every block, so it does not change the
fold. The fp32 transpose route stays unbuilt. E6 and the allocator
stay stopped. A host binary is not reviewable from this tree until
its source and the loaded library hash are recorded with the timing.

## Production fold probe (910C, 2026-10-10)

The host ran the K=128 reduce from the header this chip compiles.
ascend910_93 took `recurrent_gated_delta_rule.h`, not `arch35/`. The
entry is still `KERNEL_TYPE_AIV_ONLY`. The loaded `libcust_opapi.so`
starts `ab760fc5`. An earlier citation of that library started
`5cfdb993`; this is a later build. The recurrent kernel objects the
host names are `d3d0ef13` and `17258834`, VectorCore. The probe
source and the two benches stay on that host. They are not in this
tree.

`ReduceSumAddFoldK128` adds the two halves into `src[64]`, then one
`WholeReduceSum(dst, src[64], 64, rows, 1, 1, 16)`. The seven
arguments are destination, source, mask, repeat, destination repeat
stride, source block stride, source repeat stride. The host reads the
tiling as `vStep ≈ 64` when `dv = 128`, so production calls this with
64 rows and does a second call for the other half of V. The source
allows up to 255 rows. A probe call with 128 rows wrote 16 results.
That call is outside the production tile.

On the flush debug path, mode 14 times and mode 24 reads back. Two
tiles of 64 rows:

| shape | µs per `[128,128]` | check |
|---|---:|---|
| row `Mul`+`Add`, one-repeat `WholeReduceSum` | 7.44 | reference |
| production add-fold, two tiles of 64 | 0.98 | five families pass |

`7.44 / 0.98 = 7.6`. The families are random, zeros, tiny, a large
row, and cancellation. Under zeros, 10 of 256 heads show about 0.025
in 7 lanes. The host calls that about 1e-3 relative and suspects a
visibility race from the vector pipe into the reduce. It is not
root-caused. `0.98` µs over 128 rows is 7.7 ns/row. The row-wise form
is the extra instructions per row. After this probe, flush parity
passed, E2 bench passed, and mode 0 read 4.546 ms and 6.116 ms, the
same head1 and B=64 all-due walls as the de-scalarized binary.

Mode 15 is decode, scale, and the fold, with the casts in a row loop:
6.25 µs. `6.25 − 0.98 = 5.27` µs of those casts. A cast of 2048
elements is 32 repeats, inside the 255 limit, and the host estimates
that replacing the row loop would bring decode plus the matvec to
about 1.3 µs. That cast was not run. The same estimate applied to the
flush pass 0+1 cast loop is about 5 µs per instance, also not run.
Mode 25 read the decode path back and got NaN. Mode 24 on the same
path did not. Gate 3 has a speed and no passing correctness check.

Mode 16 times the two readouts, the compensator dots, a folded window
tile, and the output assembly, with `kn = 1` and `qn = 0.5`. It is
7.58 µs per instance. On the B=256 slowest block that is
`171 × 7.58` µs `= 1.30` ms/step, and `16 × 1.30 = 20.7` ms before
the flush. The paste's 2.9 µs, and the 0.50 ms/step that follows from
`2.9 × 171`, subtract an unmeasured cast rewrite from this 7.58 µs.
`7.58 − (6.25 − 1.3) = 2.63` µs, not 2.9. Neither 2.63 nor 2.9 was
timed. Mode 16 is a timing case, not a correctness pass.

The host's optimistic assembly, using the unmeasured cast and a flush
that has not been rewritten:

| piece | figure | what it is |
|---|---:|---|
| step, 0.50 ms, ×16 | 8.0 ms | 2.9 µs/instance, not timed |
| flush, 108 µs/instance | 18.4 ms | pass 2 cut by 32.7 µs and casts by 5, not timed |
| sum | 26.4 ms | |
| budget | 14.5 ms | |

`142 − 32.7 − 5 = 104` µs, about 4 µs under the 108 µs in that table.
`108 × 171 = 18.5` ms. `15.98 / 26.4 = 0.61`. The measured mode 16
plus the measured flush is `20.7 + 24.33 = 45.0` ms,
`15.98 / 45.0 = 0.355`. Pass 0+1, pass 3, and pass 4b are 59.3, 21.0,
and 23.2 µs. Those three are 104 µs with nothing else left, and
`104 × 171 = 17.8` ms of flush. The add-fold does not apply to that
scalar chain, to the pass 3 layout, or to the pass 4b max tree. The
0.50 ms step is also over the 0.37 ms step budget that an 8.6 ms
flush would leave. The production step is 15.98/16 = 1.00 ms.
Comparing an untimed 0.50 ms readout with that 1.00 ms leaves the
flush still in the window.

The operator line stays stopped. E5, E6, and the allocator stay
stopped. The host's reopen list is the zeros race, the chunked cast,
and a retile if the upstream dense kernel moves to a cube or MIX
path. None of those is a measured window under 14.5 ms.

## Dense live state, compressed prefix checkpoints

The quantized execution line stays stopped. A later implementation
that wants the 1.10 bar still has to time a same-machine lockstep
window at or under `15.98 / 1.10 = 14.53` ms. One split of that
budget is a step at or under 0.40 ms and an all-due flush at or under
8.0 ms (`16 × 0.40 + 8.0 = 14.4`). Those are targets. The only number
that counts is the paired window, and it is recomputed if the dense
op changes. A cube flush is not the next kernel. On this toolchain a
MIX kernel does not launch, including an empty one. An AIC-only flush
or a stream of separate kernels is the shape that can be tried, and
only after it is correct and under 8 ms.

The separate question is prefix-cache capacity with live decode left
on the dense op. A cached boundary can store the checkpoint alone:
int8 codes, fp16 compensator tokens, and the fp32 scales, 19,456 B
per head. The fp32 state of one head is 64 KiB, and
`65536 / 19456 = 3.37`. That ratio is the checkpoint against one
dense head. It is not the page line. Mode 0 is `4,243,456/1024 =
4144` B/token and mode 3 pad is `2,146,304/512 = 4192` B/token.
Shrinking the page and the block together did not change that line.
The live request in this design keeps the fp32 state and calls the
dense kernel, so the resident decode state does not shrink. The
savings are the cached copies that are not the active state. The
paper's capacity number is about one such checkpoint per 1K cached
tokens, with the high-precision window allocated once per active
request. How many of those copies this server actually holds, next to
the FA KV and the live GDN state, is not measured. Attention bytes
are still unlisted. Until that split is known, 3.37× is not a
concurrency claim, and the 1.26× startup budget is not this design.

The first measurement of that split is the section below. It stops
the capacity line for the two loads that were run. It does not reopen
E5, E6, or the allocator.

## Memory split on the dense path (910C, 2026-10-10)

Dense path only. LeapQuant off, eager, `max_model_len=8192`,
`gpu_memory_utilization=0.85`, in-process scheduler. Qwen3.5-9B. The
host script is `scripts/exp1_mem_split.py`. The raw
`logs/exp1_mem_split_{A,B}.json` files are on that host. They are not
in this tree. The layout probe that followed is
`results/migration_v2/R0/r0c_memory_ledger.json` on the same host,
also not in this tree.

Two loads, 64 requests, 256 decode tokens each, steady batch 64.

| | A, prefix cache on, align | B, prefix cache off |
|---|---|---|
| prompts | shared prefix 3584 tokens, suffix about 180 (3743–3768) | 300–349 tokens, no share |
| block table | 996 total, 263 in use | 996 total, 257 in use |
| cached-map entries | 6 | 0 |
| NPU allocated / reserved | 49.21 / 50.53 GiB | 49.21 / 50.57 GiB |

The worker sees 48 cache tensors: 24 GDN layers, each a conv bf16
view `[996,3,8192]` and an ssm fp32 view `[996,32,128,128]`. The
views sum to 47.78 GiB. Unique storage is 8 regions, 31.49 GiB
(`8 × 996 × 4,243,456`). Each storage holds one FA layer and three
GDN layers. A block is 4,243,456 B. That equals one 49,152 B span
plus two 2 MiB regions. The host text also lists a conv head and a
pad as two 49,152 B spans; those two plus two 2 MiB regions are
49,152 B over the block. The ssm view and the FA K view start at the
same address and match byte for byte on a live block. That overlap is
the hybrid layout: the 2 MiB region is KV when the block belongs to
FA, and the fp32 state when it belongs to a mamba group. The managers
do not share block ids. One sampled request had FA block 1 and mamba
group-1 block 2. The 263 in-use ids close as
`64 × (1 FA + 3 mamba) + 6 cached + 1`.

A cached entry is one block id at a 1024-token boundary, held by a
mamba group and shared across requests. It is not one copy per
sequence. The six entries are the shared prefix's boundary blocks.
Load B has none.

The pool fraction is `6 / 996 = 0.602%`. Counting each of those six
ids once in every storage gives
`6 × 8 × 4,243,456 = 0.190` GiB, 0.39% of the 49.21 GiB allocation.
The paste's 24.8 MiB is one page times six
(`6 × 4,243,456 = 24.28` MiB). The review below does not treat either
multiply as a measured byte span. Load B has no cached ids. The entry
count does not start the allocator. The 1.26× startup budget stays a
startup budget.

The paste's later cap, pool occupancy times a 1.2% GDN share, is the
49,152 B span (`49,152 / 4,243,456 = 1.16%`). That span is the conv
tail, not the fp32 state. On a mamba block the state region is 2 MiB,
49% of the block. That correction does not reopen this load. Six
block ids remain 0.60% of the table. A load of many independent long
prefixes is not measured, and it is not capped at 1.2%. The allocator
stays stopped. The compact page stays off.

## migration v2 round 1 (910C, 2026-10-10)

The host ran the first package in `migration-v2-plan.md`: R0, R1 A–C,
and the Cube codes readout. Scripts named there
(`r0c_layout_probe.py`, `r0c_alias_probe3.py`,
`bench_codes_readout.py`) are on the host. They are not in this tree.
No paired window was run.

Identity, from the host manifest: vLLM `6e448d0e`, vllm-ascend
`3b3188623` with five dirty files, opapi `ab760fc5`, CANN 9.1.0, 910C
with 37 cores. After the R1 guard edit the flush device image's first
16 hex digits are `38479a7c`. This tree's flush source still has
`NORM_EPS = 1e-30`. The 24.33 ms flush is the earlier binary, not
this image.

Servable batch, prefix cache off, prompt 4094 tokens, 256 output
tokens. B=16 and B=64 stayed resident. B=128 stayed at about 124–128
over 64 s and is the servable target. B=256 capped near 124 and waited;
it is not an end-to-end batch. It stays the operator diagnostic point.
An 8K servable batch was not in this run. The historical B=128 dense
window is 8.176 ms, and `8.176 / 1.10 = 7.433` ms. That number is not
a new pairing.

R1 A, on legal amplitudes. Zeros through the production path rebuild
to zero. Random and cancellation stay finite; the host parity is
0.01021 max relative, with its 2.4% codes-edge allowance. The host
moved the MGS floor from `1e-30` to `1e-20` and floored the absolute
sum before the scale square root, then reran parity. Scales of
`1e-24` are NaN on 256 of 256 heads, and `1e-30` on 36 of 256. The
host places that in the rebuild, upstream of the two guards. The two
iterations are spent. Amplitudes around 0.1–10 are the ones this
parity covers. R1 B: the int8 to fp16 to fp32 path is inside that
parity, `s_k` relative 1.5e-5, and the typed view matches the R0 byte
check. The debug-probe NaN stays the probe, as in the mode 25
readback. R1 C: two pools, A/B/A, pass. Re-flushing A mismatches
0.22% of codes and the median `s_k` relative error is 2.5e-4, the
same requantization noise as the parity. A and B differ on 99.4% of
the compared state. The storages differ. A reversed `idx` flushes.
R1 D, the 24-pool 96-step recycle, was not run.

R2 timed one Cube readout: int8 codes, fp32 `s_k`, real `kn`/`qn`,
exact int8 to fp16, hi/lo packed `k|q`, batched fp16 matrix multiply,
then the merge. The wall is both readouts for the step. Per-head
relative L2 against the fp32 reference:

| B | ms / step | µs / instance | max rel L2 | 3e-3 gate |
|---:|---:|---:|---:|---|
| 16 | 0.367 | 0.716 | 3.7e-4 | pass |
| 128 | 0.322 | 0.079 | 3.9e-4 | pass |
| 256 | 0.677 | 0.083 | 4.0e-4 | pass |

`0.367 / 512 = 0.717` µs, `0.322 / 4096 = 0.079` µs, and
`0.677 / 8192 = 0.083` µs. The B=16 wall is above B=128, so the small
batch is not on the per-instance line. The Vector row was not
re-timed. The 0.98 µs add-fold, the 6.25 µs decode probe, the 7.58 µs
mode 16 step, and the untimed 2.9 µs / 0.50 ms assembly stay what
they were. A sum of the Cube wall with a 0.44 ms residual is not a
step.

The readout is part of the step. At B=256 the free-flush ceiling is
`14.527 / 16 = 0.908` ms/step. The measured 0.677 ms is under that
ceiling, so this piece alone does not trip the plan's kill. It is
over the 0.40 ms step target. `16 × 0.677 = 10.832` ms, and
`14.527 − 10.832 = 3.695` ms is what remains for the compensator, the
window, the writeback, and every flush call. Comparing 10.832 ms with
the 8.127 ms column puts the readout in the non-step bucket. The
flush on the books is 24.33 ms on the earlier binary. R3 and R4 stay
closed. A B=128 reading of the old 7.433 ms gate leaves
`7.433 − 16 × 0.322 = 2.281` ms for that same remainder, and it is
not a paired dense window.

## Round-1 review (2026-10-10)

The host paste is not a raw-log replay. `main` at `55df388` has
neither the R0–R2 scripts nor the json. Three readings change.

The Cube readout stays a candidate. It does not open R3 or R4. B=256
missed the 0.40 ms step target and did not cross the free-flush kill.
`16 × 0.677 = 10.832`, and `14.527 − 10.832 = 3.695` ms remains for
the compensator, the window, the writeback, temporary tensors, the
launches, and every flush call. The 8.127 ms column is the remainder
after a 0.40 ms step. It does not cap the readout. At the historical
B=128 gate, `16 × 0.322 = 5.152` and `7.433 − 5.152 = 2.281` ms
remains for that same list. If the rest of the step were 0.10 ms,
`7.433 − 16 × (0.322 + 0.10) = 0.681` ms would be left for the flush
and the other window costs. That 0.10 ms is an illustration, not a
measurement. The untimed 2.9 µs, 0.50 ms, and 0.44 ms figures stay
out of the comparison. The batch that stayed up was about 124–128, so
this note does not treat 128 as a fixed resident batch. The next
dense pairing uses the batch that actually holds.

R1 A–C is a partial pass. `1e-24` is a normal fp32 value. Its square
underflows. The NaN is not yet pinned to subnormal inputs, and the
first non-finite intermediate is not recorded. Initialization, decay,
and cancellation can all produce a small state, so O(0.1–10) is not
a proven legal range. The 0.01021 max relative parity, and the codes
edge allowance, are not the per-head reconstruction gate on one
checkpoint plus one window. Re-flushing A and seeing requantization
noise does not prove the bind. The missing check is a frozen input,
repeated, with A's non-target bytes unchanged across the call to B.
Requantization error is a separate column. The two epsilon edits are
spent. The next record is the failure boundary, not another floor.
R2 still lacks the query readout's error, the max absolute error, a
cancellation case, and the accumulate and output dtypes. A plain fp16
path that meets the contract can stay. A hi/lo split waits on a case
that fails without it.

The memory ledger is not closed. A block described as a 49,152 B head,
two 2 MiB regions, and a 49,152 B pad sums to 4,292,608 B, which is
49,152 B over the 4,243,456 B block. One 49,152 B span plus two 2 MiB
regions does sum. The offsets are not in this tree. The live fp32
state of 64 requests, if every layer owns a private matrix, is
`64 × 24 × 32 × 128 × 128 × 4 = 3` GiB. The paste's 1.09–1.12 GiB is
257–263 block ids times one page. Those two quantities cannot be the
same footprint. A group block id applied to every layer in the group
is `6 × 8 × 4,243,456 = 0.190` GiB for the six cached ids, which is
the eight-storage multiply already written above. That multiply is a
candidate, not a measured byte span. Cold state payload, padding, and
reclaimable bytes are still separate. The 1.2% cap stays withdrawn.
Six cached-map entries against 257–263 in-use ids, and zero on the
no-cache load, still do not justify the allocator. They also do not
prove a 0.39% HBM fraction.

## Next experiments

The next package is three steps, in order.

1. Land the five dirty files, the device image, and the real
   `blockDim` so the source matches the binary. Close the ledger from
   a group block id to a storage byte range.
2. Add the missing R1 and R2 evidence. Pair dense at the batch that
   stays resident. A peak of 124–128 does not certify B=128.
3. Only after those two, time a sequential step that is the Cube
   readout plus the rest of the step, including the casts, the
   temporary global memory, and the launches. Then screen a new
   matrix flush against the remainder that step actually leaves.
   Overlap of the cube and vector cores counts only on that timeline.

R1 D is due before a full-window acceptance. E5, E6, production
wiring, MIX, the allocator, and the compact page stay off. The
servable question is the remainder at about B=128. It is not a
serving sweep.

## Still open

- The operator line is stopped. At B=256 the production window is
  15.98 ms, and R≥1.10 needs the LeapQuant window at or under 14.5 ms.
  The AscendC direct step measured on the later run is 4.253 ms, 68.0 ms
  across 16 lockstep steps. The codes matvec that actually matches the
  reference is 7.6 µs, 2.60 ms per step on the slowest block, 41.6 ms
  across 16 steps, before the window term and before the flush.
  The production-header probe on ascend910_93 passed the add-fold at
  0.98 µs per `[128,128]`, 7.6× the 7.44 µs row-wise form, on five
  input families. The earlier 1.0 µs failure was a 128-row call with
  the stride arguments in the wrong slots. Zeros still show a stale
  ~0.025 on 10 of 256 heads. The full step math that was timed is
  7.58 µs per instance, 1.30 ms/step, 20.7 ms across 16 steps, before
  flush. The 0.50 ms step and the 18.4 ms flush in the optimistic
  assembly were not timed. That sum is 26.4 ms against 14.5 ms,
  R≈0.61. The decode readback was NaN, so the chunked-cast estimate
  inside the 0.50 ms figure has no passing correctness check.
  The §15 direct-Triton row of 0.170–2.411 ms was not reproduced; the
  later Triton direct row is 2.573–8.728 ms. No multi-head step, no
  further flush polish, and no E5. E6 and the allocator stay stopped. The B=2 engine gap remains
  1.935 s step increment, 6.341 s flush region, 0.729 s prefill, and
  3.295 s unattributed. That 0.344 ms flush average is a B=2 mix, not a
  larger-batch constant. The old direct-readout kernel stays out.
  Production flags stay off. The probe kernel and benches are on the
  host and are not in this tree. The vendored-fork 0.780–0.797× scan
  is a different stack and is not re-opened here.
- The dense-path split counted 6 cached-map entries on the shared
  prefix and 0 with the cache off, against 263 and 257 in-use ids.
  The byte fraction is not closed. A private state for 64 requests is
  3 GiB, and 257–263 pages are about 1.1 GB decimal. The 1.2% cap is
  withdrawn. The allocator stays stopped. The 19,456 B checkpoint is
  still 3.37× one 64 KiB head, not the page line, and not the 1.26×
  startup budget. MIX in one kernel still does not launch. The Cube
  readout at 0.677 ms/step is not a window. R1 A–C is a partial pass.
- Step v2 wiring stays off. The 24-pool run is a state-read/write
  disagreement. Observation error and address binding are still open,
  and 0 fallbacks does not close the old E1 fault. Locate it with
  sentinels, a CPU byte copy, a device readback, and cache-off versus
  cache-on A/B/A. A separate mamba pool and the checkpoint/window split
  wait on the operator gate.
- The 96-way rate ratio is 0.54× on the reported tok/s. Completed output
  tokens are about 18,050 per side against a nominal 24,576, and the
  pasted note does not record `max_model_len` or the tokenized length.
  That ratio is not multiplied by the startup budget. A later pure-decode
  curve records the batch that ran and `T(B)` on both sides. The align
  1.26× figure remains a startup budget. The compact 1.425× row stays
  void. Page/block bytes per token are 4144 and 4192, so that one line
  does not explain the +26%. The padded page is the largest correct
  shared-layout page in the measured set.
- The container `patches/vllm-ascend-0271/` tree is an uncommitted
  74-file diff on that host. It is not a reviewable delivery in this
  repository. The multi-pool issue text is a draft, not a filed issue.
- FULL graph mode remains unsupported by the Ascend GDN backend.
- The token-27 evidence on the vendored fork points at accumulated Torch
  quantization error. A one-layer replay is optional. On that fork the
  64-token near-ties are prompts 2 and 3 only. The 0.27.1 64-token probe
  is 4/4. The port's 768-token flips at 101 and 132 are compatible with
  that quantization trajectory. An implementation difference is not
  ruled out.
- Task-level accuracy is still open. The completed AB slice is 20 prompts
  per dataset; the full five-dataset sweep is 14,634 prompts.
- `scripts/bench_capacity_graph.py` is the overnight driver. It accepts
  `--block-size`, `--no-chunked-prefill`, and an external NPU snapshot.
  That file is on the 910C host and is not in this tree.
  `scripts/bench_capacity.py` remains the eager shared-prefix script.
  Loader path selection and `smoke_qwen35.py --block-size` are already on
  `main`.
