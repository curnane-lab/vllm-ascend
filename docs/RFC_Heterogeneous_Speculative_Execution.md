# RFC: Heterogeneous Speculative Execution for Agent Serving

- **Status**: Draft (mother document)
- **Branch**: `hypic-qwen35`
- **Scope**: Unified theory and research program covering Track A (speculative
  agent execution), Track B (implicit grammar + calibrated action oracle),
  Track C (state-tree speculation as serving infrastructure), and Track D
  (branch speculation). Individual tracks should NOT re-invent evaluation
  frameworks; they inherit the definitions, utility model, and metrics here.
- **Provenance**: distilled from an extended design analysis spanning
  Hypic (arXiv:2607.01299), DFlash, ESP, PIPO, ToolSpec, MiniPIC, RedKnot,
  Bole (arXiv:2608.01651), ECHO, and the vllm/vllm-ascend codebases.

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
| **Action** | workflow grammar, action predictor | calibrated probability | wasted I/O, prefetch, bandwidth, staging |
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

Guarantee: **exact** (output distribution identical to non-speculative
decoding under the standard acceptance rule). Cost: one Target forward pass.
This is the only layer that is both exact and cheap.

### 3.2 State: Verification by Construction (Bounded Applicability Domain)

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

Any claim of exactness outside `D` is a bug. Empirically (paper
arXiv:2607.01299 + our CPU validation): linear-layer composition matches a
single-pass recompute to ~6e-5 relative norm at layer 0 (FP16 noise), fp64
~1e-16 / fp32 ~8e-8 in reference tests; end-to-end task-level deviation is
model-family dependent (near-lossless for strong-decay dense-transition
families; up to ~3–4 points for scalar slow-decay families).

### 3.3 Action: Verification by Calibration

For actions there is generally no cheap deterministic verifier; the object of
verification is the probability that the action will actually occur:

```
p_t(a) = P(A_{t+1} = a | S_≤t)
```

The system speculates only when a calibrated confidence clears a threshold:

- Acceptance set `A(S) = { a : calibrated_conf(a|S) ≥ τ }`.
- `|A(S)| = 1` → aggressive single speculation.
- `|A(S)| > 1` → multi-candidate prefetch, or branch speculation (Track D),
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

Speculate iff `U(a) > 0` with a confidence margin.

- **`L_hidden`** — latency actually removed from the critical path:
  `L_hidden = L_baseline − L_residual`. Report also the
  **Latency Hidden Ratio** `L_hidden / L_baseline`. For token/state
  speculation this reduces to saved FLOPs; for action speculation it is
  wall-clock hiding of I/O (tool, retrieval, network, prefill) — these are
  different objectives and must not be averaged into one number.
- **`C_spec`** — cost of the speculative work itself: compute, memory
  bandwidth, network, KV, retrieval, tool invocation, staging.
- **`C_resource`** — queueing/contention cost, mandatory:
  `C_resource = ΔW_q(λ, ρ) + ΔB + ΔM`, the marginal impact of speculative
  work on queue waiting, bandwidth, and memory across concurrent requests.
  This term is what flips the sign of `U` between low and high concurrency
  (the ECHO lesson): speculative prefetch is profitable at low λ and can be
  anti-profitable at high λ.
- **`C_failure`** — failure-cost class per object (table in §2), including
  rollback/staging cost for effectful actions and m× amplification for
  branches.

Branch activation condition (Track D): enable only when
`L_tool ≫ C_spec` **and** `λ < λ_threshold`, i.e. low concurrency, long tool
latency, high calibrated confidence.

---

## 5. Research Tracks

### Track A — Side-Effect-Free Speculative Execution (engineering mainline)

Pipeline:

```
predict next action → READ/SEARCH/RETRIEVE → PREFETCH → TOKENIZE
→ KV PREP → PREFILL → Target commits → reuse / discard
```

Goal: critical-path latency reduction, not FLOPs.

Experiment matrix: baseline vs. speculation across prediction accuracy,
hidden latency, speculation overhead, queueing impact (λ sweep), end-to-end
p50/p95/p99; ablations per stage (prefetch / tokenize / KV prep / prefill).
Milestone M-A1: measurable p50 TTFT/step-latency reduction on an agentic
trace with zero correctness change; M-A2: λ-threshold characterization.

### Track B — Implicit Grammar + Online Calibration (research mainline)

Pipeline:

```
trajectories → workflow mining → candidate actions
→ online/weighted conformal calibration → action set A(S)
→ speculative execution (Track A runtime)
```

Research questions:

1. Does the mined grammar generalize across tasks and tool sets (the
   overfitting failure mode of workflow mining)?
2. Does online conformal maintain coverage under distribution shift?
3. Is calibrated confidence aligned with realized latency utility?
4. Do mispredictions create systematic resource waste (C_failure audit)?

Related-work defense: workflow mining / process mining / Agent Workflow
Memory already cover trajectory→workflow induction. **Claim is NOT grammar
mining; claim is "calibrated grammar as a speculative execution oracle"** —
the middle layer (calibrated action oracle with coverage guarantees) is the
novel component.

### Track C — State-Tree Speculation as Serving Infrastructure (unique asset)

Positioning (mandatory, verbatim spirit):

> Bole (arXiv:2608.01651) establishes tree-structured closed-form state
> decomposition for speculative verification in hybrid-attention models.
> This track investigates its extension from a decoding algorithm into a
> reusable serving primitive, with explicit integration of segment-level
> PIC state reuse (`(T_C, S_{C|0})`) and NPU execution.

Non-claim: tree-structured state speculation itself. Claim: **state
speculation as serving infrastructure** — fork-and-compose GDN state at
branch points as a cheap primitive (few 128×128 matrices vs. full KV trees),
integrated with the segment cache.

First experiment (M-C1): implement a 3-branch GDN state fork on top of
`vllm_ascend/ops/hypic/state.py::compose_states`; measure branch cost vs.
acceptance gain against single-sequence drafting; verify against the
bounded-domain discipline of §3.2. M-C2: integration with segment-cache
hit states; M-C3: NPU kernel path (Triton → AscendC consistency check).

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
| Queueing impact | `ΔW_q(λ, ρ)` at swept λ | all |
| Failure cost | per-class audit (§2 table) | all |
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

Residual claim (the intersection that remains):

> **Heterogeneous verification semantics (recompute / construction /
> calibration / transaction) + calibrated action speculation with online
> conformal guarantees + resource-aware utility model + serving-level
> integration on NPU.**

---

## 8. Success Criteria

- Track A: p50 critical-path latency −≥20% on agentic trace at matched
  correctness; documented λ-threshold.
- Track B: conformal coverage within ±2% of target α under induced
  distribution shift; grammar-transfer evaluation across ≥3 task families.
- Track C: 3-branch fork cost < 10% of one Target forward per branch;
  acceptance gain vs. single-sequence draft ≥ +15% at equal budget;
  state-level parity within bounded-domain tolerance (§3.2).
- Track D: positive `U_branch` demonstrated inside the declared activation
  region, negative outside it (both results required).

---

## 9. Milestones

1. **M0 (this document)**: shared definitions, utility model, metrics.
2. **M-A1/A2**: Track A engineering validation.
3. **M-C1**: Track C fork prototype on `compose_states` (2–4 weeks est.).
4. **M-B1**: Track B calibration prototype + coverage evaluation.
5. **M-D1**: Track D conditional evaluation.
6. **M-Z**: consolidated paper draft — "Speculative Execution with
   Heterogeneous Verification".
