# B32 decode cost account and graph replay path proof

Date: 2026-10-17.
Runtime: vLLM `0.30.0`, vllm-ascend `0.30.0rc2`, CANN `9.2.0`, NPU 9.
Payload: 32 × 1024 random input tokens, fixed output tokens, prefix OFF,
seed `20261017`. PIECEWISE graph used one decode/padded capture size of 32.

This account does not replace the 0.27.1 operator window (15.98 ms /
14.527 ms). It does not accept a Q3 serving gain. The narrow
repeatable-executor binding described below was run on the host. This
commit does not contain that binding: `gdn_lq_flush_ascendc.py` still
keys the one-shot cache on `idx.data_ptr()` and `_q0_rows()` still
clears its cache on a miss.

## Graph path conclusion

Three Q3 graph runs:

- `graph1`: valid benchmark. Counter export failed on `CachedRequestData`.
- `graph2`: counters fixed. Valid benchmark.
- `graph3`: `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. Valid benchmark and the
  replay-path proof.

The server logs prove graph infrastructure, not Q3 execution:

```text
Breakable CUDA graph enabled
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
```

The replay dispatcher later logs `ACL graph replay is active`. Q3's GDN
core is invoked through `vllm::qwen_gdn_attention_core`, a custom op
decorated by `eager_break_during_capture`. With breakable graphs this is
a graph break: the operator is recorded as an eager callback on the
capture stream, and its Python body runs outside the captured graph on
every replay. The log line proves that some graph segment replays. It
does not prove that the LeapQuant quantized step executes inside a
captured graph.

EngineCore Q3 profiler counters stay empty in graph runs and increment
in eager diagnostic runs. That is consistent with Python-side execution
being excluded from graph capture.

## Cost account

Q3 PIECEWISE at 32 × 1024 input + 256 output:

| run | duration | output throughput | mean TPOT | median TTFT |
|---|---:|---:|---:|---:|
| graph1 | 48.86 s | 167.67 tok/s | 151.50 ms | 9.17 s |
| graph2 | 48.85 s | 167.68 tok/s | 151.50 ms | 9.21 s |
| graph3 | 48.72 s | 168.14 tok/s | 151.13 ms | 9.17 s |

Eager Q3 from E2A at 32 × 1024 input + 1024 output was 226.99 tok/s /
131.09 ms TPOT. The output lengths differ, so those tok/s figures are
not a ratio. TPOT is comparable:

```text
151.13 / 131.09 = 1.153x
131.09 / 151.13 = 0.867x
```

PIECEWISE is slower per decode token than that eager Q3 run. It does
not recover the Q3 gap.

## EngineCore step counters, and the withdrawn pool reading

`graph2` and `graph3` exported one JSONL row per `execute_model()`:

| phase | rows | due slots | note |
|---|---:|---|---|
| prefill/mixed | 33 | 0–98 | grows as new requests initialize |
| decode | 223 | 99–137, mean 116.77 | every decode step reports a large due count |

The first reading of 116.77 was that `pool.fill` covers all 2164
allocated state slots, so each flush of the current 32-slot batch also
folds stale composite slots and the kernel work is much larger than the
batch. That reading is withdrawn.

An independent single-pool probe used 256 slots, 32 active, inactive
`fill=16`. The flush call took about 0.08 ms. Marking only inactive
slots due did not increase it. Marking the 32 active slots due raised it
to about 4.40 ms. The AscendC kernel processes the supplied active
`idx`. Stale due markers elsewhere do not add runtime. The counter's
`due_slots` field is global pool state, not kernel work. Do not compare
116.77 with 32.

## q0-cache probe

`_q0_rows()` used to clear its cache on every miss, so 24 pools could
evict each other. The host run persists one transpose per pool. Added
counters report executor hits, misses, creations, capture rejects,
slow-path exceptions, and q0 creations. `graph4` finished with:

```json
{"hits":0,"misses":6888,"created":6888,"capture_rejects":0,
 "slowpath_exceptions":0,"q0_creates":24,"q0_evictions":0}
```

Zero hits: every graph callback runs outside the captured graph, and the
recorded eager callback recreates executors on each replay. There are no
capture rejects and no Torch fallbacks. Throughput moved from 168.14 to
168.61 tok/s, TPOT from 151.13 to 150.69 ms. The q0 cache fix is
hygiene. It does not close the Q3 gap.

## Where the 6,888 creations come from

| phase | rows | created executors |
|---|---:|---:|
| startup/mixed | 33 | 744 |
| decode | 223 | 5,352 = 223 × 24 |
| drain scheduled 31→1 | 32 | 768 = 32 × 24 |
| zero-scheduled flush step | 1 | 24 |
| total | 290 | 6,888 |

33 + 223 + 32 + 1 = 289. The following idle row creates no executor, and
289 + 1 = 290. 744 + 5,352 + 768 + 24 = 6,888. The zero-scheduled row
still creates one executor per GDN layer.

CANN 9.2 exposes `aclSetAclOpExecutorRepeatable`,
`aclSetInputTensorAddr`, and `aclDestroyAclOpExecutor` in
`libnnopbase.so`. A standalone NPU probe created one executor, marked it
repeatable, and reused it across due and not-due fills by copying new
contents into the same device `idx` buffer. No recreation. Walls
alternated at about 4.34 ms when due and about 0.05 ms when not due.
This probe is not the serving-path split. 4.34 ms is not a measured
slice of the 151.13 ms TPOT.

An `idx.data_ptr()` trace: 6,888 creations had 6,887 prior pointer
changes. `flatten().long()` changes the device buffer address, so the
one-shot cache cannot hit. The first service attempt added a repeatable
fast path but did not create the initial executor as repeatable:
`repeatable_hits=0`, and the service still made 6,888 one-shot
creations. That run is a negative control.

## Narrow-fix validation

The host binding that passed the mechanism gate:

1. caches one repeatable executor per pool/device/stride;
2. creates that executor before first execution and calls
   `aclSetAclOpExecutorRepeatable`;
3. keeps a persistent device idx buffer per batch-shape bucket;
4. copies new idx contents into that buffer on the execution stream;
5. falls back to the one-shot path when the shape changes or the
   repeatable API fails.

The 24-pool / 96-step probe passed with 24 creations and 2,304
repeatable hits (24 × 96 = 2,304). Active idx contents were rotated and
due/not-due states alternated.

The first service run hit a bucket bug: after the batch shrank from 32
to 1, the cached 32-slot idx buffer was reused for a one-slot copy and
the NPU rejected the broadcast. The fix requires the cached buffer shape
to match `idx.shape`.

Corrected no-instrumentation B32 graph run, 32/32:

| run | output throughput | mean TPOT | executor creations | repeatable hits |
|---|---:|---:|---:|---:|
| before narrow fix | 168.14 tok/s | 151.13 ms | 6,888 | 0 |
| after narrow fix | 168.36 tok/s | 150.86 ms | 504 | 6,888 |

150.86 − 151.13 = −0.27 ms. 168.36 − 168.14 = +0.22 tok/s. Decode-phase
creations dropped from 5,352 to effectively zero. The remaining 504
creations are bucket changes during startup, ramp-down, and other batch
shapes. No slow-path exceptions and no capture rejects.

## Decision

The mechanism gate passes. The performance gate fails. Removing executor
recreation and idx-address churn changed TPOT by −0.27 ms, inside run
noise, so executor construction is not the material cause of the
remaining gap.

No direct device-time split of that remaining cost was measured on this
B32 serving path. The leading hypothesis is due-flush device work, from
the standalone probe at about 4.34 ms per active B32 flush layer versus
about 0.05 ms when not due. That is a hypothesis, not an attribution.
Do not multiply 4.34 ms by 24 layers and call the product a serving
measurement.

Q3 serving stays suspended. Independent readout-kernel optimization
stays suspended. Do not proceed to step+flush fusion on this evidence.
The earlier plan to treat 116.77 due slots as extra kernel work is
withdrawn with the single-pool probe.

Working assets: the quantized-state consumer, and legal ACLNN
repeatable-executor reuse. The current Q3 serving organization has no
accepted end-to-end performance gain. This does not say that every
state-quantization approach is infeasible on Ascend 910C.

A restart needs a new computation organization with a measured budget
advantage under the same execution mode and the actual resident batch,
including the whole step, the flush, and every added overhead. Graph
safety, a successful fusion, or a correct API call are implementation
details. They are not restart criteria by themselves.
