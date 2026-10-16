# v0.30.0rc2 NPU evaluation archive

Generated: 2026-10-16.
Archive base: fork main `cb6242ec4ef18db660f916c5cb62278294f45e49`.

This note records the stack the host evaluated. It does not add a new
paired window, a memory ledger, or a task score. The patch bodies, the
vendor tarball, `evaluation-summary.md`, and `SHA256SUMS` are on that
host. They are not in this commit.

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

Relative to fork main `cb6242e`. The host checked that it applies
cleanly. It adds the GDN LeapQuant mode 0/1/2/3 integration, the
composite pool hookup, profile counters, the AscendC flush, the step-v2
hook, and a path fix. It also adds the two runtime flags and the
compact Mamba page patches. Production flags stay off until a paired
window on this stack passes its own gate. The compact page stays off
for the same reason it was withdrawn: the old compact geometry wrote
into attention V.

### `0003-local-package-base-files-add.patch`

Adds package base files copied from `vllm-ascend 0.30.0rc2`
(`__init__.py`, `logger.py`, `utils.py`, and the patch package
initializers). They exist so the research tree imports on its own under
the `PYTHONPATH` override.

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

From a clean checkout of fork main, once the host archive is present:

```bash
git checkout cb6242ec4ef18db660f916c5cb62278294f45e49
git apply patches/v030rc2-npu-eval/0001-vllm-ascend-python-port-rel-fork-main.patch
git apply patches/v030rc2-npu-eval/0003-local-package-base-files-add.patch

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
