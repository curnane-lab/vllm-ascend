# RFC: Heterogeneous Speculative Execution for Agent Serving

- **Status**: Draft v1.1 (mother document)
- **Branch**: `hypic-qwen35`
- **Scope**: Unified theory and research program covering Track A (speculative
  agent execution), Track B (utility-calibrated action oracle), Track C
  (state-tree speculation as serving infrastructure), and Track D (branch
  speculation). Individual tracks should NOT re-invent evaluation frameworks;
  they inherit the definitions, utility model, and metrics here.
- **Provenance**: distilled from an extended design analysis spanning
  Hypic (arXiv:2607.01299), DFlash, ESP, PIPO, ToolSpec, MiniPIC, RedKnot,
  Bole (arXiv:2608.01651), ECHO, and the vllm/vllm-ascend codebases.
- **v1.1 changes**: decidable applicability domain (Domain Certificate);
  utility-aware action decision rule; vectorized resource-cost model;
  perfect-future oracle experiment (M-A0) as go/no-go gate; Track B renamed
  and re-scoped; Track C re-metriced around Cached-State Forking; related-work
  boundary matrix; corrected exact/cheap claim in §3.1.

---

## 1. Problem Statement

> **When is it safe and profitable to execute the future before the Target
> has committed to it?**

Given the current system state `S_t`, a cheap oracle predicts a future
computation or execution action `a`. The system performs speculative work
before the Target commits, then applies a verification semantics appropriate
to the object being speculated, and finally commits, reuses, discards, or
rolls back.

A speculation is justified iff three conditions hold simultaneously:

```
Speculate  ⟺  Verifiable  ∧  Predictable  ∧  Profitable
```

- **Verifiable** selects the verification semantics (recompute / construction /
  calibration / transaction).
- **Predictable** selects the oracle (draft model / retrieval / FSM / grammar /
  state predictor).
- **Profitable** is decided by the utility model of §4 (hidden latency, saved
  compute, resource contention, failure cost).

Closed loop:

```
Current State
     │
     ▼
  Oracle
     │
     ▼
Speculative Work ── Token | State | Action | Branch
     │
     ▼
Verification (per-object semantics, §3)
     │
     ▼
Commit / Reuse / Discard / Rollback
```

This RFC deliberately rejects the "one bigger umbrella" style
(Speculative-Compute-OS, Compute Planner, etc.). The contribution is the
formalization of **heterogeneous verification semantics + layered utility
functions + failure-cost models**, of which each concrete mechanism is an
instance.

---

## 2. Objects of Speculation

| Object | Oracle (examples) | What is verified | Failure cost class |
|---|---|---|---|
| **Token** | DFlash, FSM, retrieval, n-gram | exact token equality | wasted draft + verification compute |
| **State** | `(T_C, S_{C|0})` composition, state fork | algebraic (by construction) | wasted state compute |
| **Action** | workflow grammar, retrieval, LLM predictor, agent memory, tool schema | calibrated probability + utility | wasted I/O, prefetch, bandwidth, staging |
| **Branch** | action predictor, top-k | calibrated probability + final state | m× resource amplification |

Branch is a **composition** of Action speculation with resource amplification
— not a fourth verification semantics.

---

## 3. Verification Semantics

### 3.1 Token: Verification by Parallel Recompute

The classical speculative-decoding contract:

```
draft tokens → Target parallel forward → exact comparison → accept longest prefix
```

Guarantee: **exact, model-level output equivalence** under the standard
acceptance rule. Cost: one Target forward pass. Token is the only layer with
exact, model-level output verification under that contract; State instead
offers construction-level correctness inside a bounded domain with zero
verification cost (§3.2). The two are different guarantee types and must not
be conflated.

### 3.2 State: Verification by Construction (Decidable Applicability Domain)

Definition. A speculative state transform `F_s` needs **no** Target
verification if it is algebraically equivalent to the Target transition `T`
inside a declared applicability domain `D`:

```
F_s(S, x) ≡ T(S, x),   ∀ (S, x) ∈ D
```

Correctness is established **by construction**, so verification cost is zero
within `D`. This is the unique layer with a *free* verification semantics —
and the RFC's central theoretical claim for the state track.

**Boundary discipline (mandatory).** Algebraic equivalence is **not**
end-to-end numerical exactness. `D` must explicitly exclude or bound:

- seam-window approximation (only the first `w` tokens of each reused segment
  are recomputed; interior cached KV deviates from full recompute);
- reused-KV deviation in full-attention layers of hybrid stacks;
- deep-layer hidden-state drift compounding through the stack;
- kernel/precision mismatches (e.g., Triton vs. AscendC extraction paths;
  fp32 staging discipline).

Any claim of exactness outside `D` is a bug.

**Domain Certificate (v1.1).** `D` is not a disclaimer; it is a decidable
predicate. Decompose:

```
D = D_architecture ∩ D_cache ∩ D_precision ∩ D_sequence
```

with machine-checkable membership conditions, e.g.:

- `D_architecture`: layer is a supported linear-attention variant (GDN/KDA/
  scalar family); transition-operator family known.
- `D_cache`: segment was persisted with the required tuple `(T_C, S_{C|0},
  conv_tail)`; seam width `w` within validated range; hash verified
  (`seg_hash` + token-id recheck).
- `D_precision`: states staged and stored in fp32; kernel inputs in model
  native dtype; extraction path cross-validated (Triton vs. AscendC).
- `D_sequence`: composition applied in segment position order; per-branch
  state forked from a certified parent state.

Most conditions are static at startup (architecture, precision, kernel path);
the remainder are checked per segment at cache-hit time. **Every composed or
forked state must carry a domain certificate** (`domain_valid ∈ {true, false}`
plus the failing predicate) in logs and diagnostics; a false certificate
forces fallback to recompute. This turns "verification by construction" from
a paper statement into a system primitive.

Empirical anchors (paper arXiv:2607.01299 + our CPU validation): linear-layer
composition matches a single-pass recompute to ~6e-5 relative norm at layer 0
(FP16 noise); fp64 ~1e-16 / fp32 ~8e-8 in reference tests; end-to-end
task-level deviation is model-family dependent (near-lossless for
strong-decay dense-transition families; up to ~3–4 points for scalar
slow-decay families).

### 3.3 Action: Verification by Calibration — and by Utility

For actions there is generally no cheap deterministic verifier; the object of
verification is the probability that the action will actually occur:

```
p_t(a) = P(A_{t+1} = a | S_≤t)
```

**Coverage is not utility (v1.1).** A conformal guarantee
`P(A_{t+1} ∈ A_t) ≥ 1−α` only says the true action is probably in the set;
it says nothing about whether speculating on it is profitable (a 100%-certain
1 ms action with a 3 ms prefetch cost is a loss; a 65% 500 ms action with a
20 ms cost may be a win). The action-layer decision rule is therefore:

```
Speculate(a)  ⟺  CalibratedRisk(a) ≤ ε  ∧  E[ U(a) | S_t ] > 0
```

i.e., calibrated risk **and** positive expected utility under the model of
§4 — conformal calibration and the utility gate are joint conditions, not
sequential stages.

Operationalization via the acceptance set:

- `A(S) = { a : calibrated_conf(a|S) ≥ τ }`.
- `|A(S)| = 1` → aggressive single speculation (subject to the rule above).
- `|A(S)| > 1` → multi-candidate prefetch, branch speculation (Track D),
  or abstain.

**Hard requirement: online / weighted conformal calibration.** Vanilla
conformal prediction assumes exchangeability; agent trajectories are
non-stationary and strongly autocorrelated (tooling updates, user drift,
long-range dependencies), so vanilla coverage guarantees degrade
systematically in deployment. Track B adopts online conformal with adaptive
quantiles and time-decayed sample weighting as its core theoretical component.

### 3.4 Side-Effect Discipline (Transaction Semantics)

Verification semantics interact with side effects:

| Operation class | Speculation mode |
|---|---|
| Pure / idempotent (read, retrieve, search, tokenize, KV prep, prefill) | execute ahead; reuse if hit, **discard** if wrong |
| Effectful but stageable (file edit, local compute) | sandbox / shadow copy → **commit/rollback** (cf. Cursor Shadow Workspace) |
| Effectful and external (send, submit, deploy, delete, git push) | **no speculation** in v1 |

Non-claim: speculative edit with staging already exists (Cursor Shadow
Workspace). The claim is the *generalization* to a serving-runtime primitive
with a side-effect-level semantics table.

---

## 4. Utility Model

Per-speculation net utility:

```
U(a) = L_hidden(a) − C_spec(a) − C_resource(a) − C_failure(a)
```

Speculate iff `U(a) > 0` (jointly with the action-layer rule of §3.3 where
applicable).

- **`L_hidden`** — latency actually removed from the critical path:
  `L_hidden = L_baseline − L_residual`. Report also the
  **Latency Hidden Ratio** `L_hidden / L_baseline`. For token/state
  speculation this reduces to saved FLOPs; for action speculation it is
  wall-clock hiding of I/O (tool, retrieval, network, prefill) — these are
  different objectives and must not be averaged into one number.
- **`C_spec`** — cost of the speculative work itself: compute, memory
  bandwidth, network, KV, retrieval, tool invocation, staging.
- **`C_resource`** (v1.1, vectorized) — contention cost with explicit unit
  conversion (time, bandwidth, and memory are NOT summed raw):

  ```
  C_resource = c_q·ΔW_q + c_b·ΔBW + c_m·ΔMEM + c_n·ΔNET = cᵀΔR
  ```

  `ΔW_q` = marginal queueing delay added to co-running requests;
  `ΔBW`, `ΔMEM`, `ΔNET` = marginal bandwidth, memory-occupancy, and network
  consumption. Coefficients `c_q, c_b, c_m, c_n` convert each resource to a
  common latency-equivalent cost and are calibrated per deployment (they are
  policy knobs, not constants of nature). This term flips the sign of `U`
  between low and high concurrency (the ECHO lesson): speculative prefetch is
  profitable at low λ and can be anti-profitable at high λ.
- **`C_failure`** — failure-cost class per object (table in §2), including
  rollback/staging cost for effectful actions and m× amplification for
  branches.

Branch activation condition (Track D): enable only when
`L_tool ≫ C_spec` **and** `λ < λ_threshold`, i.e. low concurrency, long tool
latency, high calibrated confidence.

---

## 5. Research Tracks

Execution order is **A → C → B → D**: first prove money is on the table,
then build on unique assets, then invest in prediction, then amplification.

### Track A — Side-Effect-Free Speculative Execution (engineering mainline)

Pipeline:

```
predict next action → READ/SEARCH/RETRIEVE → PREFETCH → TOKENIZE
→ KV PREP → PREFILL → Target commits → reuse / discard
```

Goal: critical-path latency reduction, not FLOPs.

**M-A0 (go/no-go gate): Perfect-Future Oracle Benchmark.** Before building
any predictor, replay a complete agent trace with the *ground-truth* future
action as the oracle, and compare four arms: perfect-oracle prefetch /
predictor-arm / random-prefetch / no-speculation. Measure Latency Hidden
Ratio, queueing penalty (λ sweep), and end-to-end p50/p95/p99. If the
perfect oracle hides only a few percent of the critical path, Tracks B/D are
de-prioritized regardless of predictor quality; if it hides 30–50%, the
program is validated for investment. No grammar, no conformal, no training
in this phase.

M-A1: measurable p50 TTFT/step-latency reduction on an agentic trace with a
real predictor and zero correctness change. M-A2: λ-threshold
characterization (`cᵀΔR` calibration).

### Track B — Utility-Calibrated Action Oracle (research mainline)

Pipeline:

```
oracle sources (workflow grammar / retrieval / LLM predictor /
agent memory / tool schema)
→ calibrated action oracle (online/weighted conformal)
→ utility gate E[U]>0 (§3.3, §4)
→ speculative execution (Track A runtime)
```

Grammar is one oracle source among several; the named component is the
oracle, not the mining.

Research questions:

1. Does the oracle generalize across tasks and tool sets (the overfitting
   failure mode of workflow mining)?
2. Does online conformal maintain coverage under distribution shift?
3. Is calibrated confidence aligned with realized utility
   (`E[U]>0` precision/recall)?
4. Do mispredictions create systematic resource waste (`C_failure` audit)?

Related-work defense: workflow mining / process mining / Agent Workflow
Memory already cover trajectory→workflow induction. **Claim is NOT grammar
mining; claim is a utility-calibrated action oracle as a speculative
execution primitive** — the joint calibration×utility middle layer is the
novel component.

### Track C — Cached-State Forking (unique asset)

Positioning (mandatory, verbatim spirit):

> Bole (arXiv:2608.01651) establishes tree-structured closed-form state
> decomposition for speculative verification in hybrid-attention models.
> This track investigates its extension from a decoding algorithm into a
> reusable serving primitive, with explicit integration of segment-level
> PIC state reuse (`(T_C, S_{C|0})`) and NPU execution.

The named primitive is **Cached-State Forking**: the segment cache supplies
certified parent states (`S0` from PIC reuse), and state fork composes
multiple futures from them:

```
                 Segment Cache (certified parent state S0)
                    │
              ┌─────┴─────┐
              ▼           ▼
        compose(A)   compose(B)
              │           │
             S1          S2
```

This is why the work belongs to Hypic/PIC serving rather than being "another
Bole-style decode algorithm": PIC provides *cached states*, state tree
provides *forking from cached states*. Non-claim: tree-structured state
speculation itself.

**Metrics (v1.1, re-scoped).** Acceptance gain is demoted to an intermediate
diagnostic; the primitive metrics are:

- **State Fork Cost Ratio** = `C_fork+compose / C_full state recompute`;
- **Branch State Coverage** = candidate future states maintainable per fixed
  state budget;
- **End-to-end utility** = final arbiter (p50/p95 latency, `U` per §4).

M-C1: implement a 3-branch GDN state fork on top of
`vllm_ascend/ops/hypic/state.py::compose_states`; measure the two primitive
metrics and fork overhead; verify domain certificates (§3.2) hold.
M-C2: integration with segment-cache hit states. M-C3: NPU kernel path
(Triton → AscendC consistency check).

### Track D — Branch Speculation (conditional, last)

Mandatory distinction for review:

```
Search      = evaluate multiple branches, select best
Speculation = predict one/few likely branches, execute ahead, commit/discard
```

Enable only under the activation condition of §4. Out of scope for v1:
effectful branches; high-concurrency regimes.

---

## 6. Metrics (shared across tracks)

| Metric | Definition | Layer |
|---|---|---|
| Acceptance / coverage | accepted fraction; conformal empirical coverage vs. target α | token / action |
| Latency Hidden Ratio | `L_hidden / L_baseline` | action / branch |
| FLOPs saved | baseline − actual (per generated token) | token / state |
| Overhead ratio | `C_spec / L_baseline` | all |
| Queueing impact | `cᵀΔR` at swept λ | all |
| Failure cost | per-class audit (§2 table) | all |
| State Fork Cost Ratio | `C_fork+compose / C_full recompute` | state |
| Branch State Coverage | future states per fixed state budget | state |
| End-to-end | p50 / p95 / p99 TTFT and per-step latency | system |

No track may report acceptance rate as a headline metric without the
corresponding utility.

---

## 7. Related Work and Non-Claims

| Work | Already established | We do NOT claim |
|---|---|---|
| ToolSpec (arXiv:2604.13519) | schema/retrieval → token draft | first structural-prior draft |
| AWM / process mining | trajectory → workflow induction | first workflow mining |
| Cursor Shadow Workspace | side-effect staging for edits | first speculative edit |
| OpenAI Predicted Outputs | external draft → generation speedup | first draft reuse |
| LLMCompiler | planner → parallel tool execution | first early tool execution |
| Bole (arXiv:2608.01651) | tree state decomposition | first state-tree speculation |
| Hypic / PIC line | context/KV/state reuse | first KV reuse |
| ECHO | verification cost under concurrency | first contention analysis |
| MCTS / ToT | multi-branch search | (Track D is speculation, not search) |

Boundary matrix (v1.1) — the residual intersection is four-dimensional:

| | Predict | Execute ahead | Verify | Resource-aware |
|---|---|---|---|---|
| ToolSpec | ✓ | ✗ | ✓ | ✗ |
| LLMCompiler | ✓ | ✓ | planner-level | limited |
| Cursor | ✓ | ✓ | staging | ✗ |
| Bole | ✓ | ✓ | construction | ✗ |
| Hypic/PIC | ✗ | ✓ | construction-ish | ✓ |
| **Ours** | ✓ | ✓ | **heterogeneous** | **✓** |

Residual claim:

> **Heterogeneous verification semantics (recompute / construction /
> calibration / transaction) + utility-calibrated action speculation with
> online conformal guarantees + resource-aware utility model + serving-level
> integration on NPU.**

---

## 8. Success Criteria

- Track A: M-A0 oracle study quantifies the latency ceiling; M-A1 achieves
  p50 critical-path latency −≥20% on agentic trace at matched correctness;
  documented λ-threshold.
- Track B: conformal coverage within ±2% of target α under induced
  distribution shift; oracle-transfer evaluation across ≥3 task families;
  `E[U]>0` precision/recall reported.
- Track C: State Fork Cost Ratio ≪ 1 with Branch State Coverage ≥3;
  domain certificates hold on all composed/forked states; end-to-end utility
  positive at equal budget.
- Track D: positive `U_branch` demonstrated inside the declared activation
  region, negative outside it (both results required).

---

## 9. Milestones

1. **M0 (this document)**: shared definitions, utility model, metrics.
2. **M-A0**: perfect-future oracle benchmark (go/no-go gate).
3. **M-C1**: Cached-State Forking prototype on `compose_states`.
4. **M-A1/A2**: Track A predictor integration + λ calibration.
5. **M-B1**: Track B utility-calibrated oracle prototype + coverage
   evaluation.
6. **M-D1**: Track D conditional evaluation.
7. **M-Z**: consolidated paper draft — "Speculative Execution with
   Heterogeneous Verification".
