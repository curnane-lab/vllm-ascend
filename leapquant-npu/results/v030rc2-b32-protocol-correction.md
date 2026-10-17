# Why the earlier mode3 result looked neutral but the rerun is much slower

Date: 2026-10-17.

The two numbers are not the same experiment. The earlier “B32” result
launched 4 short prompts with 256 output tokens. The rerun launches 32
requests with 128 fixed output tokens. In the old run, 32 was
`max_num_seqs`, the scheduler cap, not a measured resident batch. Most
of that wall time is low-batch decode. The rerun is the closer estimate
of a full-B32 steady workload, and it is where mode3 overhead shows up.

This note does not replace the 15.98 ms operator gate, and it does not
close the five-pair acceptance.

## Raw comparison

| setup | requests | requested/output tokens | wall | throughput |
|---|---:|---:|---:|---:|
| old mode0 “B32” | 4 | 256 | 13.01 s | 78.73 tok/s |
| old mode3 “B32” | 4 | 256 | 13.22 s | 77.48 tok/s |
| new U0 B32 | 32 | 128 fixed | 5.98 s | 684.69 tok/s |
| new Q3 B32 | 32 | 128 fixed | 9.50 s | 431.06 tok/s |

Token totals if every request emits its cap: old 4 × 256 = 1024, new
32 × 128 = 4096.

Reported throughput against those totals: 1024 / 13.01 = 78.71 tok/s
(table 78.73), 1024 / 13.22 = 77.46 tok/s (table 77.48),
4096 / 684.69 = 5.982 s (table wall 5.98 s), 4096 / 431.06 = 9.502 s
(table wall 9.50 s). The table rates are the recorded measurements.
77.48 / 78.73 = 0.984 on this one old pair.

The host’s round load comparison is `1024 / 13.0 ≈ 79` tok/s against
`4096 / 6.0 ≈ 685` tok/s. The reported rates give
684.69 / 78.73 = 8.70×. These walls are not one steady-state batch
measured twice.

## Mechanism

- Old run: `requests=4`, `max_num_seqs=32`, `max_num_batched_tokens=8192`,
  `max_model_len=4096`. Only 4 prompts are launched. Most iterations
  contain one decode token (B=1), then a short overlap window. “B32” is
  the scheduler upper bound.
- New run: 32 prompts are launched, so B32 is the intended resident
  batch.
- New run adds `ignore_eos=True` and `min_tokens=max_tokens=128`. Every
  request runs to exactly 128 tokens.
- The old run did not suppress EOS. Ramp and drain can hide mode3
  step/flush cost.
- In the new run the host reports a mode3 geometry change: attention
  block 640→384, padding 22.14%→43.28%, LCM block 5120→3072. That is
  this v0.30 rerun, not the 0.27.1 page table (attn block 1024, mode3
  pad page with attn block 512). A different cache/page path and
  scheduler cost is plausible. No profiler counter yet accounts for the
  whole gap.

## Current B32 medians

| config | valid blocks | throughput tok/s | host paired ratio to U0 |
|---|---:|---:|---:|
| U0 | 4 | 684.29 | 1.000x |
| R0 | 4 | 679.04 | 0.982x |
| Q3 | 3 | 444.38 | 0.650x |
| QV2 | 3 | 417.05 | 0.600x |

Ratio of these median throughputs, which is a different statistic from
the paired-ratio column: 679.04 / 684.29 = 0.992, 444.38 / 684.29 =
0.649, 417.05 / 684.29 = 0.609. The per-block pairs are not in this
commit, so 0.982 and 0.600 cannot be recomputed here. Q3 and QV2 have
3 valid blocks; U0 and R0 have 4. This is not five complete pairs.

Block 1 alone was U0 684.69, R0 679.94, Q3 431.06, QV2 403.97 tok/s.
The medians above are not that single block.

The Q3/QV2 gap that remains after separating the 4-request run from the
32-request run is a real finding. It still needs worker-level profiler
counters and five complete paired blocks before an engineering
decision. QV2 stays `requested step-v2` until those counters show it
ran.

The 2026-10-17 graph account in
`leapquant-npu/results/v030rc2-b32-graph-cost.md` suspends Q3 serving.
The 4-request check below was not that account.

## Immediate corrective next step

Keep the new fixed-length payload and launch only 4 requests with
`max_num_seqs=4`. That should sit closer to the old short-prompt regime
and check whether mode3 overhead shrinks as the resident batch shrinks.
Then run the five-block B32 comparison and export EngineCore profiler
counters.
