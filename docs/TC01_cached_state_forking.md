# TC01: Cached-State Forking — Technical Design

- **Status**: Design v0.9 (pre-implementation)
- **Track**: C (Cached-State Forking), RFC §5
- **Depends on**: RFC v1.2 (`docs/RFC_Heterogeneous_Speculative_Execution.md`),
  `vllm_ascend/ops/hypic/state.py` (compose_states), `transition.py`
  (T_C extraction), `SGLANG_REFERENCE_SPEC.md`
- **Goal of this document**: pin down the fork semantics, algebra, memory
  model, integration points, and validation protocol before writing code.

---

## 1. The technical point in one paragraph

A GDN (linear-attention) recurrent state is a small set of matrices
(per layer per head: `S ∈ R^{K×V}` fp32, plus a K−1 conv tail). Because the
recurrence is affine in the state, `S_out = T_C·S_in + S_{C|0}`, **forking a
state costs a matrix copy, and advancing a fork costs one matmul per cached
segment** — independent of segment length. A full-attention model has no such
primitive: branching a context there means maintaining a KV tree whose size
grows with context length. Cached-State Forking (CSF) turns this asymmetry
into a serving primitive: maintain many independent future evolutions from
one certified parent state at near-zero marginal cost per branch.

This is the unique asset of the RFC: it is only available on
hybrid/linear-attention models, only usable because the `(T_C, S_{C|0})`
composition algebra exists (validated, `state.py`), and it has no published
serving-side precedent (Bole uses tree state decomposition as a decode-time
verification algorithm, not as a serving primitive; see RFC §5 Track C for
the mandatory positioning).

---

## 2. Semantics

### 2.1 Objects

- **Parent state** `P`: a certified GDN state bundle =
  `{ S[ℓ,h], conv_tail[ℓ], domain_certificate }` for all linear layers
  `ℓ ∈ L_gdn`, heads `h`. Sources: (a) PIC segment cache hit, (b) live
  prefix prefill of a running request.
- **Branch** `B_i`: a handle referencing `P` plus its own advance history.
  A branch is *lazy*: it owns no state until first advanced (copy-on-write).

### 2.2 Operations

| Op | Meaning | Cost |
|---|---|---|
| `fork(P) → {B_1..B_k}` | create k branch handles on P | O(1) per branch (handle + refcount on P) |
| `advance(B_i, seg)` | cached segment: `S ← T_seg·S + S_{seg\|0}` | `O(L·H·K²·V)` per segment, length-independent |
| `advance(B_i, tokens)` | novel tokens: replay GDN recurrence from current state | normal prefill cost of those tokens |
| `commit(B_i)` | promote branch state to the request's running (decode) slot | O(1) (slot handover) |
| `discard(B_i)` | release branch slots; decrement parent refcount | O(1) |

Invariants:

1. A parent cannot be evicted while `refcount(P) > 0` (branches hold locks,
   same discipline as SGLang HYPIC's `lock_ref`, see SGLANG_REFERENCE_SPEC §8).
2. Every branch state carries a domain certificate derived from the parent's
   certificate plus the branch's own inputs (RFC §3.2). A branch whose
   inputs fall outside `D` must be marked `domain_valid=false` and falls back
   to recompute.
3. Composition order is always segment position order within a branch
   (RFC §3.2, `D_sequence`).

### 2.3 What forking is NOT

- Not a KV tree (SpecInfer-style): no per-token tree bookkeeping, no tree
  attention mask. State is a point, not a path.
- Not a new verification algorithm: branch correctness comes from
  verification-by-construction (RFC §3.2), not from re-checking against the
  Target.
- Not free advance for novel tokens: replay costs the same as prefill. CSF
  only amortizes the *shared prefix* and the *cached segments*.

---

## 3. Algebra and cost model

### 3.1 Fork consistency (the correctness claim)

For branch `B` advanced from parent `S_P` over segments `C_1..C_n`:

```
S_B = T_n·...·T_1·S_P + Σ_i (Π_{j>i} T_j)·S_{C_i|0}
```

which is exactly `compose_states(S_P, T_{1..n}, S_{1..n|0})`. Because
composition is associative-consistent with the token-level recurrence
(validated: fp64 ~1e-16, fp32 ~8e-8 vs. token-level unroll), the branch state
equals a sequential recompute of the same segment sequence from `S_P`, up to
staging precision. **Fork consistency test = this equality, per layer.**

### 3.2 Cost accounting (Qwen3.5-9B as reference)

Assumptions: 24 GDN layers, H≈16 heads/layer, K=V=128, fp32 states.

- State size per request: `24 × 16 × 128 × 128 × 4B ≈ 25 MB`.
- Fork: O(1) handle (copy-on-write); materialization `≤ 25 MB` per branch
  only on first advance.
- Advance by cached segment: `24 × 16 × 128² × 128 × 2 FLOP ≈ 1.6 GFLOP`
  ≪ one Target forward for the same segment (for a 1k-token segment,
  ~2 orders of magnitude cheaper; exact ratio to be measured as
  **State Fork Cost Ratio**, RFC §6).
- Advance by novel tokens: same as prefill (no savings, no overhead).

Branch State Coverage at 64 GB HBM with a 10 GB state pool:
`⌊10 GB / 25 MB⌋ ≈ 400` materialized branch states per model instance
(before attention KV and pool overheads). The metric is reported per model
configuration (state shape is family-relative, RFC v1.2).

### 3.3 Where the asymmetry comes from

| | Full-attention branch | GDN branch (CSF) |
|---|---|---|
| shared-prefix artifact | KV tree, grows with context | parent state, fixed size |
| branch advance (cached span) | re-read KV, attention compute | one matmul per layer |
| branch advance (novel span) | prefill | prefill (identical) |
| memory per extra branch | O(context × layers) | O(25 MB) fixed |

The advantage is structural, not incremental: it comes from the recurrence
compressing history into a fixed-size state. It therefore persists at any
context length, and **grows** with context length.

---

## 4. Use cases (priority-ordered)

### UC1 — RL rollout prefix sharing (first application)

RL post-training (GRPO/OPD-style) draws N rollouts per prompt. All N share
the prompt prefix; today each rollout re-prefills or shares KV only for
full-attention layers. For hybrid models, CSF completes the sharing:

```
prefill prompt once → parent state P
fork N branch states → advance each with its own sampled continuation
```

- Full-attention KV sharing is already handled by prefix/radix caching;
  **the GDN state is the missing half**, and CSF is exactly that half.
- Fits the Draft-OPD training stack directly: rollout engines are
  SGLang/vLLM-ascend; the state pool is per worker.
- Measurable win: prefill FLOPs per prompt ÷ N for the GDN part; end-to-end
  rollout wall-clock reduction on long prompts (agentic RL, tool-use RL
  with 10k+ token contexts).

This is the recommended first application because the utility is measurable
without any predictor: the branches are real rollouts, not speculations.

### UC2 — Agent branch speculation (Track D feeder)

Fork at decision points, advance top-k candidate continuations (agent
actions), commit the branch the Target selects, discard the rest. UC2 is the
feeder of Track D's branch speculation; it requires UC1's machinery plus the
calibrated action oracle (Track B), so it is sequenced after UC1.

### UC3 — Session forking in agent frameworks

Parallel task branches from a shared long context (repo state, memory) in
coding agents. Same machinery; product-side rather than training-side.

---

## 5. Implementation plan

### P0 — Math and BranchSet (CPU-verifiable, ~1 week)

In `vllm_ascend/ops/hypic/state.py` (no vllm imports, CPU-testable):

- `StateBundle`: per-layer `{S, conv_tail}` + certificate dict.
- `BranchSet`: `fork(parent) → branch_ids`; `advance(branch, transitions,
  states)` via existing `compose_states`; `advance_tokens(branch, ...)` as an
  interface stub for the NPU replay path; `commit/discard` with parent
  refcounting; per-branch certificate propagation with the §2.2 invariant 2.
- Tests (`tests/ut/ops/test_hypic_state.py` additions):
  - fork consistency: fork+advance == sequential unroll from parent
    (dense and GDN-form transitions, fp64/fp32, k ∈ {1,3,8});
  - refcount: parent locked until all branches discarded;
  - certificate: out-of-domain branch inputs mark `domain_valid=false`.

### P1 — NPU integration (~2–3 weeks)

- MambaPool: multi-slot allocation per request for branch states (extend
  `_alloc_one_mamba`-equivalent in vllm-ascend); parent slots shared
  read-only, branch slots per-branch.
- Replay path: GDN decode/prefill kernel with per-branch `initial_state`
  (hook exists, `gdn.py:691` in v0.21.0rc1).
- Conv-tail inheritance per branch (layout per SGLANG_REFERENCE_SPEC §5).
- Lock discipline: branch/parent lock_ref in the pool's eviction path.

### P2 — UC1 pilot: rollout prefix sharing (~2 weeks)

- Minimal scheduler change: recognize "shared-prompt rollout groups"
  (same prompt hash, N continuations) → one parent prefill + N forks.
- Metrics: GDN prefill FLOPs per group, rollout wall-clock vs. baseline,
  pool pressure (Branch State Coverage under load).

---

## 6. Validation protocol

| Metric | Target | Method |
|---|---|---|
| Fork consistency | rel. err < 1e-5 (fp32 staging) | P0 CPU test vs. token-level unroll |
| State Fork Cost Ratio | ≪ 1 (expect ~1e-2 for 1k-token segments) | P1 measurement, per segment length sweep |
| Branch State Coverage | ≥ 8 branches/instance at Qwen3.5-9B | P1 pool pressure test |
| UC1 prefill FLOPs/group | ≈ baseline/N for GDN part | P2 pilot |
| UC1 wall-clock | measurable reduction at long prompt | P2 pilot, p50/p95 |
| Domain certificates | 100% present, false→fallback | all phases (RFC §3.2) |

Kill criteria (pre-registered): fork consistency fails beyond staging
tolerance; fork+compose is not ≥10× cheaper than recompute at realistic
segment lengths; pool pressure makes coverage <3 at target concurrency.

---

## 7. Risks and open questions

1. **Conv-tail semantics on fork**: the K−1 conv tail must be inherited
   per branch and then diverge; cheap (tiny), but a correctness-critical
   detail (SGLANG_REFERENCE_SPEC §5 layout differences gdn/kda).
2. **MoE layers interleaved**: branch replay passes through full-attention
   layers too — the full-attention side needs its own sharing story
   (prefix cache) for UC1 to deliver end-to-end; CSF alone only covers the
   GDN half. The pilot must report both halves.
3. **Pool fragmentation**: many small branch slots vs. large segment slots;
   mitigation = separate sub-pool for branches (echoes SGLang HYPIC's
   public/private pool split).
4. **bf16 replay vs fp32 composition**: branch states materialized from
   composition are fp32; the replay kernel consumes bf16 — cast policy must
   match the boundary discipline (fp32 staging, RFC §3.2 `D_precision`).
5. Open: whether branch states should be evictable to CPU/DDR under
   pressure (probably v2; interacts with offload connectors).

---

## 8. Relationship to the RFC

This document operationalizes RFC §5 Track C and §3.2. It does not modify
the RFC. When P0–P2 data exists, the RFC's Track C success criteria
(State Fork Cost Ratio, Branch State Coverage, end-to-end utility) are
evaluated against §6 here and the RFC is updated to v1.3 with measured
numbers replacing targets.
