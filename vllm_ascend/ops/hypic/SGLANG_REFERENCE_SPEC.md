# SGLang HYPIC → hypic-qwen35 (vllm-ascend): Reference Semantic Spec

Reverse-engineered from the SGLang HYPIC fork at `E:/work/ref/HYPIC`
(single commit `98147c019`, "Hypic code on sglang v0.5.14"). The design
document (`qianyou/2026-05-28-pic-sglang-design.md`) is **referenced but not
shipped** in that repo — this spec is extracted from code only, with
`file:line` evidence. Purpose: implementation-neutral semantics that the
vllm-ascend port must reproduce, plus the mapping to this repo's modules.

Scope note: this document pins **semantics** (what must hold), not mechanisms
(how SGLang does it). Where our port deliberately diverges (e.g. MiniPIC-style
position-free K instead of re-rotation), the divergence is recorded and the
oracle protocol below defines how equivalence is judged.

---

## 1. Segment model

| Rule | SGLang evidence | hypic-qwen35 status |
|---|---|---|
| Prompt split by separator (default `<<PIC_SEP>>`, flag `--pic-separator-str`); separator never enters the token stream | `segmenter.py:14-34`, `server_args.py:1868-1871` | planned (M2): segment metadata via InputProcessor |
| Each segment tokenized independently (`add_special_tokens=False`); empty segments dropped, no offset emitted | `segmenter.py:28-30` | same |
| Segment endpoints are indices into the concatenated token sequence | `segmenter.py:33-34` | same |
| **The last segment (query) is never hit, never cached** | `picache.py:56-57, 275-276` | must keep |
| First segment never seam-recomputed | `picache.py:291-294` | must keep |
| Optional min-tokens filter (`pic_segment_min_tokens`, -1 = off). WARNING: filtering a segment also breaks conv-tail chaining for its successor (prev slot = -1 → conv1d from zero state) — filtering changes numerics, not just cache size | `picache.py:277-279`, `conv_tails.py:56-62` | defer; if implemented, inherit the numeric side effect |

## 2. Content addressing

| Rule | SGLang evidence | hypic-qwen35 status |
|---|---|---|
| `seg_hash = sha256(token_ids as int32 little-endian bytes)[:16]` (128 bit) | `segmenter.py:37-49` | adopt same digest for cross-stack cache sharing |
| Hash is only an index key; a hit requires shape match AND `torch.equal` token-by-token recheck | `picache.py:193-203` | must keep |
| Idempotent insert: on hash collision the **late caller frees its own allocated slots** | `picache.py:379-384`, `scatter_xfer.py:391-396` | must keep |
| Scatter path uses a separate FNV-1a 64-bit `text_hash` that must byte-match the Rust gateway (`sgl-model-gateway::policies::pic::text_hash`) | `scatter_xfer.py:61-71` | N/A until segment parallelism (M8) |

## 3. Per-segment cache content (per layer)

| Object | SGLang semantics | hypic-qwen35 status |
|---|---|---|
| Full-attention KV | token-level slots (their `page_size=1` constraint exists because `full_kv_slots` has one entry per token) | we keep block tables (vLLM); v1 constraint **not** inherited |
| GDN state `S` | fp32, in MambaPool slot; one slot per segment regardless of length | `state.py SegmentCache.zero_start_state` ✓ |
| Transition `T_C` | fp32, `[layers, size+1, H_v, K, K]` pool allocated only for transition-family modes | `transition.py extract_segment_transitions` ✓ |
| Conv tail | last K-1 raw `mixed_qkv` (pre-conv) per segment, stored per mamba slot; per-segment conv slots (shared per-request slot caused a fancy-index race — fixed upstream); gdn layout `[D, K-1]` (transposed on store), kda `[K-1, D]`; zero-init required for segments shorter than K-1 | `SegmentCache.conv_state` field ✓, capture/load logic **not yet implemented** |

## 4. Composition law (transition family)

SGLang (right-multiply convention, states stored `[H_v, V, K]`):

```
h <- bmm(h, T[slot]) + S[slot]     # applied in segment position order
```

Ours (left-multiply convention, states `[..., K, V]`, from the FLA kernel):

```
S <- T_C @ S + S_{C|0}             # compose_states(), state.py
```

**The two `T_C`s are transposes of each other.** Within each stack the
convention is self-consistent; cross-stack oracle comparisons must transpose.
Both are validated: theirs in production code, ours against token-level
unroll (fp64 ~1e-16).

Composition ordering rules (must keep):
- Strict segment position order; miss-segment starts asserted monotonic (`gdn_backend.py:605-620`).
- A miss segment's Pass-3 replay is seeded with the composed state **before** its own advance (snapshot-then-advance, `gdn_backend.py:1377-1383`).
- Final composed state is written to the request's running (decode) slot.
- `addition` mode (sum of hit S only, seeding the first miss segment) exists but has a declared weakness with multiple miss segments (`gdn_backend.py:1129-1131`) — we do not plan to implement it.

## 5. Seam window (recompute mode)

| Rule | SGLang evidence | hypic-qwen35 status |
|---|---|---|
| Width `w` default 8 (`PIC_SEAM_SINK`); `0<w≤1` = fraction of segment, `w>1` = tokens, `w≤0` = off | `pic/__init__.py:8-18` | same default w=8 (paper) |
| `w_eff = min(w, seg_len - 1)` — at least one interior token always remains pooled | `picache.py:291-294` | must keep |
| Hit segments (except the first) exclude the first `w_eff` tokens' KV from the prefix and recompute them in-request; the pool stores interior-only (S,T) for cold-prefilled segments | `picache.py:267-300`, `gdn_backend.py:963-968` | M5 (piecewise prefill), not started |
| Seam K/V overwrite the hit segment's private seam slots **after** rerotate (phase order A→B→C is mandatory) | `hybrid_linear_attn_backend.py:1503-1523, 1722-1724` | applies if we adopt dual-slot RoPE; see §6 |

## 6. Full-attention KV: the RoPE question (deliberate divergence)

SGLang HYPIC (`transition_rope`) uses **dual slots**: public slot holds K
de-rotated to pos=0 (this is what enters the cache entry); private slot holds
real-position K; on hit, public K is re-rotated by the segment start into the
request's private slots; miss segments are deliberately **not** re-rotated
(avoids bf16 round-trip drift, `hybrid_linear_attn_backend.py:1640-1645`).
Partial rotary: only the first `rotary_dim` dims are rotated.

hypic-qwen35 instead adopts the **MiniPIC/LazyAttention design**: cache
unrotated K̃ and rotate inside the attention kernel per request
(`K_tile · R(π(j))`). Rationale: no private-copy storage amplification, no
per-hit re-rotation pass, composes with offload. Trade-off: requires an
Ascend attention kernel with in-kernel RoPE (FIA support unlikely → Triton).

**Oracle consequence**: cross-stack byte-level KV comparison is meaningless
(the cached bytes legitimately differ). Equivalence is judged at the output
level — see §9.

## 7. Numerical requirements (hard-won, must inherit)

1. **fp32 staging everywhere for states**: S, T, h_accum staging buffers and
   the pool are fp32 (`mamba_ssm_dtype=float32`). bf16 S costs ~1e-3 per
   composed segment, ~3e-3 end-to-end (`gdn_backend.py:741-759`).
2. **w/u kernel inputs stay bf16**: their Triton 3.6.0 produced garbage with
   fp32 `tl.dot` in the v-reconstruction (`gdn_backend.py:1315-1320`). This
   specific bug is Triton-version-specific, but the general rule — *keep the
   h-recurrence inputs in the model's native dtype, states in fp32* — carries.
3. **Same kernel family for prefill and decode** of linear layers (SGLang
   forces `triton` for both, `server_args.py:6684-6699`): state S and
   transition T must be numerically consistent between whoever produces and
   whoever consumes them. NPU equivalent: **the AscendC production operator
   and our Triton extraction path must be cross-validated** (already on our
   README risk list) — or T_C extraction must move into the same AscendC op.
4. Batch token order == absolute position order across `input_ids`,
   `positions`, `out_cache_loc`, and rope metadata — four sites must agree
   (`pic_alloc.py:250-266` etc.). Our M5 piecewise-prefill design has the
   same invariant on the vLLM side.

## 8. Lifecycle & allocation (reference for our scheduler work)

- `lock_ref` is multi-source: match does not lock; transition-family insert
  locks +1 until request end; scatter has three independent holds
  (write-pin / pending-hold / combine-hold), each with exactly one release
  point. Missing any leaks mamba slots → OOM (`scatter_xfer.py:269-286`).
- Release blind spots SGLang had to patch explicitly: decode-generated KV
  slots, rope private slots (both miss and hit copies), last-segment slots,
  and the PD-decode role's whole `[0, committed)` range
  (`picache.py:438-556`). Our design must enumerate equivalent paths in the
  vLLM refcount model.
- Slot tensors entering requests/entries must be **cloned**, never views of
  allocator buffers (`picache.py:219-227`).
- Eviction: dual resource requirement (tokens AND mamba slots), LRU over
  entries, candidates are `lock_ref == 0` only (`picache.py:570-593`).
- Their known-accepted defect: concurrent combines sharing an entry may
  under-decrement refcount (v1 accepted). We should not accept this in ours.

## 9. Oracle / validation protocol (agreed with the SGLang line)

1. **Within-stack**: PIC-on vs PIC-off output parity per mode
   (their `diag_layer_dump.py` + `tools/diag_layer_diff.py` workflow: dump
   per-layer fingerprints, find the first diverging layer; replicate with our
   own dumps).
2. **Cross-stack (SGLang ↔ vllm-ascend)**: compare at the **output level**
   (greedy logits / generated tokens on a fixed prompt suite), plus task
   metrics per the paper's protocol. Never compare raw cached bytes (RoPE
   schema diverges by design, §6) or raw states without transposing (§4).
3. **Numeric gates**: composed-state relative error vs single-pass recompute
   ≤ fp16 noise (~1e-4 rel norm, cf. paper §4.2); our composition kernel
   already validated at fp64 ~1e-16 / fp32 ~8e-8.
4. Dev vehicle: **Qwen3.5-9B** (dense hybrid, 32 layers, no MoE confounds) —
   cheap inner loop before 35B-A3B.

## 10. Known traps in the SGLang implementation (do not copy)

- `check_pic_constraints` references `mamba_scheduler_strategy`, but the real
  field is `mamba_radix_cache_strategy` — the guard is likely dead code
  (`server_args.py:6710` vs `:307-312`).
- The Bailing/Ring `cross_segment` attention variant's docstring contradicts
  its code (public slot stores real-pos K, hit copies without re-rotation).
  Qwen path unaffected; do not use it as the spec source
  (`hybrid_linear_attn_backend.py:1182-1287`).
- `hybrid_linear_attn_backend.py:1372-1408` docstring describes an obsolete
  Phase-A design; current code defers miss-segment queries to Phase C.
- KDA transition buffers assume `head_k_dim == head_v_dim`
  (`memory_pool.py:414-421`) — check before enabling KDA models.
- `pic_mode` help text "only addition is implemented" is stale
  (`server_args.py` params block); all four modes are implemented.

---

*Extracted by code reading (subagent + review), 2026-07. Cross-check against
the paper (arXiv:2607.01299) is the SGLang line's responsibility per the
agreed division of labor.*
