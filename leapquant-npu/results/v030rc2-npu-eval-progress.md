# v0.30.0rc2 NPU port progress

## Scope and base

This archive records the NPU port and evaluation work relative to fork main at
`cb6242e` (exact commit in `README.md`). The patches are intended for manual
upstream synchronization.

## Completed work

### Runtime port and validation

- Rebased the local research port onto installed vLLM `0.30.0` and
  vllm-ascend `0.30.0rc2`.
- Made the local source overlay independently importable and added the
  package-base files copied from vllm-ascend `0.30.0rc2`.
- Preserved dense mode0, mode3 Triton/AscendC, and mode3v2 three-op step
  execution paths behind `VLLM_ASCEND_GDN_LEAPQUANT` and
  `VLLM_ASCEND_GDN_LQ_STEP_V2`.
- Verified 48-token smoke outputs: mode0, mode3, and mode3v2 dumps were
  byte-identical.
- Verified 768-token generation: 2/2 prompts token-identical for mode3 and
  mode3v2 versus mode0.
- Verified 200 GSM8K plus 200 MATH-500 greedy questions: all configurations
  scored 95.50% and 77.50%, and all 400 generated token sequences matched
  mode0.
- Corrected the local benchmark model default to
  `/data1/models/Qwen3.5-9B`.

### P0 repeatability and system behavior

- Completed 27/27 eager throughput points: three modes, batches 8/16/32, and
  three runs.
- Reported mode3 relative medians 1.006x, 0.973x, and 0.990x under the
  labels batches 8, 16, and 32. The batch-32 label is not a measured
  resident batch: the 2026-10-17 correction shows that setup launched 4
  prompts with `max_num_seqs=32`. One raw pair is mode0 78.73 tok/s
  (13.01 s) and mode3 77.48 tok/s (13.22 s), ratio 0.984. See
  `leapquant-npu/results/v030rc2-b32-protocol-correction.md`. Batches 8
  and 16 are not re-described there.
- Isolated a reproducible mode3v2 eager batch16 slow tail: median 0.856x,
  CV 8.9%. Batch8 and batch32 remain normal.
- Verified piecewise graph mode at batch16 and 128 tokens: mode0 596.9,
  mode3 605.2, and mode3v2 599.8 tok/s. Graph replay removes the mode3v2
  tail, pointing to launch/executor scheduling rather than mathematical
  error.
- Probed compact pages with `--mamba-cache-mode none`: the CLI accepts the
  flag, but the v0.30rc2 integration does not activate. Geometry and KV
  capacity remain identical between mode0 and mode3.
- Attempted in-engine profile export with `VLLM_ASCEND_GDN_LQ_PROFILE=1`.
  Counters remain in EngineCore and do not reach the driver-side JSONL, so
  tail attribution needs an IPC/export patch.

## Archived evidence

Primary evidence lives in the repository result directory:

- `results/v030rc2_npu_eval/summary.md`
- `results/v030rc2_npu_eval/e2e/e2e_summary.md`
- `results/v030rc2_npu_eval/e2e/p0/p0_interim_report.md`
- `results/v030rc2_npu_eval/e2e/next_experiment_plan.md`

## Upstream notes

- No additional C++ patch is required relative to fork main: the fork already
  contains `vllm-ascend/csrc/attention/gdn_lq_*` sources. The runtime uses the
  archived prebuilt custom-op vendor package.
- The standalone Triton path currently requires
  `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`; replacing the global `QMAX` with a
  constexpr is a follow-up.
- The archive intentionally excludes bulky result JSONL files and generated
  caches; result summaries and key evidence remain in the results directory.

## Validation v1, B32 block 1 (2026-10-16)

Logical NPU 8. One eager block, prefix off, 32 × 128 output tokens.
This is not the five-pair acceptance and not a production pass.
Details: `leapquant-npu/results/v030rc2-validation-v1-b32.md`.

Offline `LLM.generate` (driver batch, not a service baseline): U0
684.69, R0 679.94 (0.993x, −0.69%, inside the 2% screen), Q3 431.06
(0.630x), QV2 403.97 (0.590x) tok/s. The earlier neutral “B32” mode3
number is the 4-request run above, not this resident batch. Later
medians, with unequal valid blocks, are U0 684.29 (4), R0 679.04 (4),
Q3 444.38 (3), QV2 417.05 (3) tok/s. The host paired ratios are
0.982x / 0.650x / 0.600x; the ratios of those medians are 0.992, 0.649,
and 0.609. Online HTTP block 1: U0 432.92, R0 433.90
tok/s (1.002x), 32/32 each. Q3 and QV2 were not in that service block.
QV2 is requested step-v2; EngineCore did not export `v2_status()` into
the benchmark JSONL, so execution is unproven. Identity currently
proves R0 import at startup from the driver, not the worker map.

## Next priorities

1. B-equivalence: keep the fixed-length 128-token payload, launch 4
   requests with `max_num_seqs=4`, and see whether mode3 overhead
   shrinks with the resident batch. Then finish five complete paired
   B32 blocks and export EngineCore profiler counters before deciding.
2. Move identity collection into the EngineCore worker and export
   `v2_status()` through an IPC/file mechanism. Do not interpret the
   B32 QV2 rate until per-worker step-v2/fallback counters are exported.
3. The offline 684.69 tok/s row is not the service baseline. Service
   block 1 is U0 432.92 and R0 433.90 tok/s.
4. Add an EngineCore-to-driver counter export path and reproduce the batch16
   mode3v2 tail with direct counters.
5. Investigate the v0.30rc2 compact-page hook and enable actual unequal-page
   geometry and capacity.
6. Run full GSM8K and MATH-500 greedy regression.
7. Continue OpenAI-server/async-client benchmarking past this one HTTP
   block, then production trace replay. The block-1 service rates are
   U0 432.92 and R0 433.90 tok/s only.
8. Fix the Triton constexpr dependency. Step-v2 executor and input-pool
   optimization wait until a block shows that step v2 actually ran. The
   target remains mode3v2 at least matching mode3 at batches 16 and 32.
