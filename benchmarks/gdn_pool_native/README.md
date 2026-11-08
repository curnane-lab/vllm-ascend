# Native GDN pool prefill experiment

This directory contains the Ascend C `ChunkGatedDeltaRuleFwdPool` operator and its Stable-ABI adapter used by `VLLM_ASCEND_GDN_POOL_IO=1`. It is an opt-in experiment, not an upstream-ready operator package. A5 is untested, the formal FLA ATK package is not included, and the scalar implementation has not shown a microbenchmark speedup.

The pool entry reads BF16 `[V,K]` states from source slots and writes destination slots directly. Q/K/V are `[B,HK/HV,T,128]`; g and beta are `[B,T,HV]`. Chunk metadata uses chunk size 64. vLLM preserves every temporal cache copy, supplies running source/destination slots without host readback, and does not retry a failed native launch on a possibly modified pool.

## Build

Prerequisites:

- An Ascend 910B environment with CANN 9.2, Python 3.12, `torch_npu`, `g++`, and `gawk`.
- A `flash-linear-attention-npu` source checkout at commit `a4a7958`, including its submodules: <https://github.com/flashserve/flash-linear-attention-npu>.
- A complete original `fla_npu_transformer` vendor package from the matching FLA installation. The two-operator pool package does not contain unrelated operators such as causal convolution.

The installer copies the operator and adapter into that checkout, builds both the production forward and pool operators, and creates an isolated runtime under `OUTPUT_DIR`. The opapi overlay resolves pool symbols first and uses the original complete library only for missing symbols.

The `operator/` and `runtime/` subtrees are third-party source snapshots. They are excluded from this repository's automatic reformatting and spelling hooks so those hooks cannot rewrite the tested implementation. They are instead covered by the build and numerical checks below. The experiment's own adapters, scripts, and documentation remain subject to the normal hooks.

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu.git
cd flash-linear-attention-npu
git checkout a4a7958
git submodule update --init --recursive

bash benchmarks/gdn_pool_native/install_operator.sh \
  "$PWD" \
  /path/to/gdn_pool_output \
  /path/to/original/fla_npu_transformer
```

The original complete vendor directory contains `op_api/lib/libcust_opapi.so`. Build logs and artifact hashes are written to `OUTPUT_DIR/build.log`, `OUTPUT_DIR/stable_build.log`, and `OUTPUT_DIR/artifacts.sha256`.

## Runtime

Use a fresh process after setting the paths. `accessor` is required for the validated experiment because the default no-wait Stable-ABI path did not produce deterministic final states in the direct production-FLA control. This is a common setting for both A/B arms, not a pool optimization.

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/path/to/gdn_pool_output/runtime:${PYTHONPATH}
export FLA_NPU_STABLE_LIB=/path/to/gdn_pool_output/runtime/libfla_npu_stable.so
export FLA_NPU_OP_API_LIB=/path/to/gdn_pool_output/runtime/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so
export FLA_NPU_STABLE_STREAM=accessor
export ASCEND_CUSTOM_OPP_PATH=/path/to/gdn_pool_output/runtime/fla_npu/opp/vendors/fla_npu_transformer:/path/to/gdn_pool_output/runtime/full_vendor
export VLLM_ASCEND_GDN_POOL_IO=1
export VLLM_ASCEND_GDN_POOL_LAYOUT_FUSION=0
```

Use BF16 Mamba SSM cache. Unsupported layouts, filtered-sequence metadata, missing operators, or disabled pool I/O return to the existing production path before a native launch. Numerical and serving evidence is recorded separately under `benchmarks/results`; it does not establish upstream merge readiness.
