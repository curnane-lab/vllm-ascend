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
- Confirmed mode3 is production-neutral: relative medians 1.006x, 0.973x, and
  0.990x at batches 8, 16, and 32.
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

## Next priorities

1. Add an EngineCore-to-driver counter export path and reproduce the batch16
   mode3v2 tail with direct counters.
2. Investigate the v0.30rc2 compact-page hook and enable actual unequal-page
   geometry and capacity.
3. Run full GSM8K and MATH-500 greedy regression.
4. Add OpenAI-server/async-client benchmarking and production trace replay.
5. Fix the Triton constexpr dependency and continue step-v2 executor and
   input-pool optimization; target mode3v2 at least matching mode3 at batches
   16 and 32.
