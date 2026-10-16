# v0.30.0rc2 NPU evaluation archive

Generated: 2026-10-16.
Archive base: fork main `cb6242ec4ef18db660f916c5cb62278294f45e49`.

This note records the stack the host evaluated. It does not add a new
paired window, a memory ledger, or a task score.
`0001`, `0003`, and `0004` are in `patches/v030rc2-npu-eval/`. The
vendor tarball, `evaluation-summary.md`, and `SHA256SUMS` are still
only on the host.

## Runtime versions

| component | version / source |
|---|---|
| vLLM | `0.30.0` (`0.30.0+empty`) |
| vllm-ascend | `0.30.0rc2`, source commit `8fde6d4e335fe300e2b949aab0e31659cce0811d` |
| torch_npu | `2.10.0.post7` |
| torch | `2.10.0+cpu` |
| CANN | `9.2.0` |
| device | Ascend910, logical device 15 |
| model | Qwen3.5-9B |

The installed `vllm-ascend` tree is read-only. Evaluation imports this
repository's `vllm-ascend/` package by putting it first on `PYTHONPATH`.
Earlier 0.27.1 / CANN 9.1.0 rows in `M5-measurement-update.md` are a
different stack. Their 15.98 ms dense window is not this stack's gate
until it is measured again.

## Patches

### `0001-vllm-ascend-python-port-rel-fork-main.patch`

Relative to fork main `cb6242e`. `git apply --check` on that tree
succeeds. The host paste searched for `libcust_opapi.so` under a
personal absolute path. The stored patch deletes the
fork's `/mnt/workspace/vllm-ascend/...` glob and searches
`_cann_ops_custom` beside the package
(`os.path.join(os.path.dirname(__file__), "..", "_cann_ops_custom", ...)`).
The two binding hunks are `@@ -33,7 +33,8 @@` and
`@@ -32,7 +32,8 @@`, matching the seven context-and-removal lines in
`cb6242e`.

The diff sits on top of the LeapQuant code already in `cb6242e`. It
does not add mode 0/1/2/3, the composite pool, the AscendC flush, step
v2, or the compact page. Those stay where they already are, and the
compact page stays off: the old compact geometry wrote into attention V.

Three changes, all default off:

1. `_LQProfiler` in `gdn.py`, entered only when
   `VLLM_ASCEND_GDN_LQ_PROFILE` is set. Each measured span synchronizes
   the device on entry and exit. The flush span also reads a due count
   with `.item()`. Leave the variable unset.
2. The package-relative `libcust_opapi.so` search in
   `gdn_lq_flush_ascendc.py` and `gdn_lq_step_ascendc.py`.
3. A Triton branch in `triton/gdn_leapquant_step.py` behind
   `VLLM_ASCEND_GDN_LQ_DIRECT_READOUT` (default `"0"`) and
   `VLLM_ASCEND_GDN_LQ_DIRECT_VBLOCK` (default `16`). Leave it off. It
   is the same listing already rejected under "Direct-readout step" in
   `M5-measurement-update.md`: the codes term divides by `QMAX` twice,
   the compensator `q`/`u` terms are not loaded, a length-`VBLOCK`
   partial is added to a length-`V` vector, and the shared epilogue
   reads `w` that this branch never loads. The comment says
   compensators are included; the branch does not touch `u_ptr` or
   `q_ptr`. This is not the R2 Cube readout.

### `0003-local-package-base-files-add.patch`

Adds package base files copied from `vllm-ascend 0.30.0rc2`
(`__init__.py`, `logger.py`, `utils.py`, and the patch package
initializers). They exist so the research tree imports on its own under
the `PYTHONPATH` override.

### `0004-local-evaluation-scripts-rel-fork-main.patch`

Relative to fork main `cb6242e`. `git apply --check` on that tree
succeeds. Five files: `bench_e2e.py` adds the `sys` import and moves
the local model default to `/data1/models/Qwen3.5-9B`,
`test_step_triton.py` gains a `TRITON_STEP_MODE` dense/direct switch
that prefers the AscendC flush kernel with a torch fallback, and three
evaluator scripts land under `tmp/` with the host's scratch names — an
E2E benchmark with request metrics (`tmp/F0BtVT.py`), the GSM8K +
MATH-500 greedy evaluator against reference answers (`tmp/ZIeo68.py`),
and the 8198-token single-prompt per-mode profiler (`tmp/ebkGtx.py`).
The dataset and reference-answer defaults inside `tmp/ZIeo68.py` point
at host-only paths (`/data1/f00538480/...`, `/data1/w00842045/...`)
and are arguments, not contracts. The evaluation numbers themselves
are in `v030rc2-npu-eval-progress.md`.

### `0002-ascendc-csrc-ops-rel-fork-main.patch`

Not emitted. Every tracked `csrc/attention/gdn_lq_*` source is already
in fork main `cb6242e`. The only local difference the host reported is
the file mode, `100644` to `100755`.

### `custom_transformer-vendor-20261016.tar.gz`

Runtime AscendC vendor package used by the evaluation. It unpacks to
`vllm_ascend/_cann_ops_custom/vendors/custom_transformer` and contains
the prebuilt ACLNN APIs, metadata, and kernels for `gdn_lq_*`. The
archive is not in this commit, so this tree cannot name the loaded
kernel hash.

## Apply

`0001` applies on a clean checkout of fork main:

```bash
git checkout cb6242ec4ef18db660f916c5cb62278294f45e49
git apply patches/v030rc2-npu-eval/0001-vllm-ascend-python-port-rel-fork-main.patch
```

`0003` applies on the same checkout:

```bash
git apply patches/v030rc2-npu-eval/0003-local-package-base-files-add.patch
```

`0004` applies there too, and brings the evaluation scripts:

```bash
git apply patches/v030rc2-npu-eval/0004-local-evaluation-scripts-rel-fork-main.patch
```

The vendor tarball is not in this tree. When the host archive is
present, the rest of that checkout is:

```bash
mkdir -p vllm-ascend/vllm_ascend/_cann_ops_custom/vendors
tar -xzf patches/v030rc2-npu-eval/custom_transformer-vendor-20261016.tar.gz \
  -C vllm-ascend/vllm_ascend/_cann_ops_custom/vendors
```

Point `PYTHONPATH` at this repository's `vllm-ascend` directory, and
point `ASCEND_CUSTOM_OPP_PATH` at the unpacked `custom_transformer`
directory. Those two paths are machine-local and are not recorded here.

The standalone Triton path currently needs
`TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`. The Triton runtime rejects the
kernel's global `QMAX`. That remains a compatibility TODO.
