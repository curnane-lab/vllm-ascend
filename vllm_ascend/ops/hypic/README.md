# Hypic: position-independent caching for hybrid-attention models (Qwen3.5)

This package ports the model-side primitives of **Hypic** (arXiv:2607.01299)
to vllm-ascend, targeting Qwen3.5-style GDN hybrid stacks.

**Reference spec**: `SGLANG_REFERENCE_SPEC.md` — implementation-neutral
semantics reverse-engineered from the SGLang HYPIC fork, the mapping to this
package, and the cross-stack oracle/validation protocol.

## What is here (increment 1)

| Module | Content | Status |
|---|---|---|
| `state.py` | `SegmentCache` tuple `(T_C, S_{C\|0}, conv_state)` and the composition law `S = T_C @ S_prefix + S_{C\|0}` (Eq. 6) | Done, CPU-validated (`tests/ut/ops/test_hypic_state.py`) |
| `transition.py` | `extract_segment_transitions`: T_C extraction by re-running the chunked GDN h-recurrence with `h0 = I`, `u = 0` over the existing Triton FLA kernel (`ops/triton/fla/chunk_delta_h.py`) | Done, NPU test included |

Key facts these build on (verified against this codebase, v0.21.0rc1):

- `chunk_gated_delta_rule_fwd_h` already accepts `initial_state` /
  `output_final_state`; with `u = 0` and `h0 = I` its final state is exactly
  `T_C` (the recurrence is linear in the incoming state; `w`, `g` do not
  depend on `v`).
- Per-call inputs `(k, w, g)` come from the standard pre-processing chain
  (`chunk_local_cumsum` → `chunk_scaled_dot_kkt_fwd` → `solve_tril` →
  `recompute_w_u_fwd`), so transition extraction reuses tensors the prefill
  path already computes.

## Roadmap (not in this increment)

- Segment-level cache pools (public/private, LRU) keyed by segment token-id
  hash — bypasses vLLM's chained prefix block-hash by design.
- Seam-window (w = 8) piecewise prefill: per-segment "pseudo sequences" in
  `GDNAttentionMetadataBuilder` + `slot_mapping`-driven KV rewrite in the
  full-attention backend (`AscendAttentionBackend`).
- 0-based RoPE caching of K + re-rotation at reuse
  (`patch_qwen3_5.py` currently RoPEs with absolute positions before cache).
- Segment parallelism across instances via the Mooncake connector
  (mamba conv/ssm state transfer already exists; T_C needs buffer
  registration).
- Numerics: Triton extraction path must be cross-validated against the
  production AscendC operator `torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h`.

## Notes

- Qwen3.5 (GDN, dense transition family) needs no state RoPE re-rotation;
  scalar-family models (Ring-2.5) must re-rotate `S_{C|0}` before composing
  (see `compose_states` docstring).
- T_C extraction currently requires head dim K ≤ 128 (h-kernel block layout).
- The h-kernel always stores per-chunk intermediate states `h [B, NT, H, K, K]`;
  in transition mode these are unused (transient waste of ~NT × H × 32 KB per
  segment). A dedicated kernel or a `store_intermediate=False` variant is
  future work.
- Integration into the production prefill path (`chunk_gated_delta_rule_fwd`,
  which already computes `(k, w, g)` via the WY chain) is the next increment;
  `extract_segment_transitions` deliberately reuses those tensors rather than
  recomputing them.
