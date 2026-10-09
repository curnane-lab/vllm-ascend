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
which is 41.9 / 47.85 ≈ 0.876× dense. Flush is the first target, and
flush alone does not pass dense on this run.

The 0.344 ms average also mixes every call. It does not separate a
nothing-due launch, a partial fold, an all-due fold, or a cold bind.
The next split, on a synchronized window, records for nothing-due: call
count, host submit time, device time; for mixed and all-due: due-slot
count, call count, device time; for the window: 16-step total, fold
count, and a state check. A controlled upper bound calls flush only when
a slot is known due. Production still has to cover async requests, slot
recycle, and restore, so a global step modulo is not that bound.
Flush v2 t3 stays a candidate on the experimental chain. It is not the
next production target. The order on this gap is the flush v1 call path,
then the step increment, then the unattributed residual.

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

## Still open

- Next on the 0.27.1 eager gap: record the timer's per-region counts,
  including the last partial 4096-bucket, and split flush v1 into
  nothing-due, mixed-due, and all-due on a synchronized window. The
  flush timer region is 6.341 s of the 12.3 s gap (51.6%). The step
  increment is 1.935 s (15.7%). The remaining 3.295 s (26.8%) is
  unattributed. Zeroing the measured flush time leaves about 0.876×
  dense, so flush is necessary and not sufficient. Flush v2 t3 stays a
  candidate on the experimental chain only. Direct-readout still waits
  on a kernel that matches the materializing formula. The B=8 bench
  remains a mixed wall. Production flags stay off. The vendored-fork
  0.780–0.797× scan is a different stack and is not re-opened here.
- Next on step v2: the 24-pool run narrowed the failure to a
  state-read/write disagreement. Observation error and address binding
  are still open, and 0 fallbacks does not close the old E1 fault.
  Locate it with sentinels, a CPU byte copy, a device readback, and
  cache-off versus cache-on A/B/A. Wiring stays off. A separate mamba
  pool stays deferred until this and the flush split have answers.
- The 96-way rate ratio is 0.54× on the reported tok/s. Completed output
  tokens are about 18,050 per side against a nominal 24,576, and the
  pasted note does not record `max_model_len` or the tokenized length.
  The next capacity points use one cache and page configuration, record
  waiting, running, TTFT, TPOT, completions, and output tokens, and sit
  between the two admission boundaries. "Admission-bound plus light
  decode" is a hypothesis. The align 1.26× figure remains a startup
  budget. The compact 1.425× row stays void. The padded page is the
  largest correct shared-layout page in the measured set; that bound
  does not replace the serving check.
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
