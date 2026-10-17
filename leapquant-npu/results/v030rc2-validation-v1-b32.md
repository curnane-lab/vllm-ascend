# v0.30.0rc2 validation v1 — executed B32 block 1

Date: 2026-10-16.
Scope correction: 2026-10-17.
Device: logical NPU 8.
Runtime: vLLM `0.30.0`, vllm-ascend `0.30.0rc2`.
Payload: 32 requests × 128 fixed output tokens, eager, prefix OFF,
`ignore_eos=True`.

This block is not a production acceptance, and it is not the 0.27.1
operator window (15.98 ms / 14.527 ms). That window is a different
stack.

## Result

| config | throughput | vs U0 | note |
|---|---:|---:|---|
| U0 | 684.69 tok/s | 1.000x | clean installed upstream tree |
| R0 | 679.94 tok/s | 0.993x | patched mode0, within 2% screen |
| Q3 | 431.06 tok/s | 0.630x | research mode3 |
| QV2 | 403.97 tok/s | 0.590x | requested step-v2 |

Checks: 679.94 / 684.69 = 0.993, 431.06 / 684.69 = 0.630,
403.97 / 684.69 = 0.590. (679.94 − 684.69) / 684.69 = −0.69%.
32 × 128 = 4096. All four runs completed 4096 generated tokens with
`fixed_length=true`.

**Scope correction (2026-10-17):** these four rates came from an offline
`LLM.generate` driver, not from a vllm-ascend HTTP service. They are
driver-side batch results and are not a service baseline. Online
vllm-ascend HTTP comparisons use the host's `server_b32/` artifacts.

Online service baseline, block 1 only: U0 432.92 tok/s, R0 433.90 tok/s
(433.90 / 432.92 = 1.002x), 32/32 successful requests each. Q3 and QV2
have no row in that service block. The host cites `server_b32/README.md`
and `server_b32/summary.json`. Those two files are not in this commit.

## Interpretation and limits

- The offline U0/R0 difference is −0.69%. This block does not breach the
  2% porting-overhead screen. The online pair is 1.002x on its own
  protocol. Neither pair is the five-block acceptance set.
- The offline Q3/QV2 gap is not the earlier “B32” mode3 result. That
  earlier run launched 4 prompts of 256 tokens under `max_num_seqs=32`;
  32 was the cap, and one raw pair is 78.73 vs 77.48 tok/s. The 1024
  in `1024 / 13.0` is 4 × 256, not this block’s payload. This block is
  32 × 128 fixed tokens. The protocol split is
  `leapquant-npu/results/v030rc2-b32-protocol-correction.md`. After that
  split, the Q3/QV2 gap is a real finding and still needs counters plus
  five complete pairs.
- This is one block, not the five-pair acceptance set.
- The identity artifact currently proves the full R0 tree import path at
  NPU runtime startup. It is a driver-side proxy, not the final
  EngineCore worker process map.
- QV2 status counters were added at source level but were not exported
  from the EngineCore process into the benchmark JSONL. This QV2
  throughput is `requested step-v2`. Actual v2 execution is still
  unproven. Do not read 403.97 tok/s as a measured step-v2 pass.

## Immediate next work

1. B-equivalence check: same fixed-length payload, 4 requests,
   `max_num_seqs=4`, then the five-block B32 comparison.
2. Move identity collection into the EngineCore worker and export
   `v2_status()` through an IPC/file mechanism.
3. Capture four more paired blocks to form the five-block U0/R0 result.
4. Export per-worker step-v2/fallback counters before interpreting QV2.
5. Re-run after identity closes. Do not use this block as production
   acceptance.
