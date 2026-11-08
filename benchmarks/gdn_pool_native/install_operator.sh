#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 FLA_NPU_SOURCE_DIR OUTPUT_DIR FULL_VENDOR_DIR" >&2
    exit 2
fi

source_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
full_vendor_dir=$(realpath "$3")
asset_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runtime_dir="$output_dir/runtime"
vendor_dir="$runtime_dir/fla_npu/opp/vendors/fla_npu_transformer"
full_runtime_vendor="$runtime_dir/full_vendor"
operator_dir=fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_pool

if [[ ! -f "$source_dir/build.sh" || ! -f "$source_dir/torch_custom/fla_npu/csrc/build_stable.py" ]]; then
    echo "FLA_NPU_SOURCE_DIR is not a complete flash-linear-attention-npu checkout" >&2
    exit 1
fi
if [[ ! -f "$full_vendor_dir/op_api/lib/libcust_opapi.so" ]]; then
    echo "FULL_VENDOR_DIR must contain the original complete fla_npu_transformer vendor package" >&2
    exit 1
fi
command -v gawk >/dev/null
command -v g++ >/dev/null

mkdir -p "$output_dir" "$(dirname "$source_dir/$operator_dir")"
cp -a "$asset_dir/operator/chunk_gated_delta_rule_fwd_pool" "$source_dir/$operator_dir/.."
cp -a "$asset_dir/runtime/torch_custom/fla_npu/." "$source_dir/torch_custom/fla_npu/"

source "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash}"
cd "$source_dir"
bash build.sh -j32 --soc=ascend910b --pkg --vendor_name=fla_npu \
    --ops=chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_fwd_pool \
    >"$output_dir/build.log" 2>&1
./build_out/fla_npu_linux-aarch64.run --quiet --install-path="$output_dir/opp"

mkdir -p "$vendor_dir"
cp -a "$source_dir/torch_custom/fla_npu/fla_npu" "$runtime_dir/"
cp -a "$output_dir/opp/vendors/fla_npu_transformer/." "$vendor_dir/"
mkdir -p "$full_runtime_vendor"
cp -a "$full_vendor_dir/." "$full_runtime_vendor/"

op_api_dir="$vendor_dir/op_api/lib"
pool_op_api="$op_api_dir/libcust_pool_opapi.so"
cp "$op_api_dir/libcust_opapi.so" "$pool_op_api"
full_op_api="$full_runtime_vendor/op_api/lib/libcust_opapi.so"
g++ -shared -fPIC "$asset_dir/opapi_overlay.c" \
    -Wl,--no-as-needed,"$pool_op_api","$full_op_api" \
    -o "$op_api_dir/libcust_opapi.so"
python "$source_dir/torch_custom/fla_npu/csrc/build_stable.py" \
    --out "$runtime_dir/libfla_npu_stable.so" --no-debug-probe \
    >"$output_dir/stable_build.log" 2>&1

export PYTHONPATH="$runtime_dir:${PYTHONPATH:-}"
export FLA_NPU_OP_API_LIB="$op_api_dir/libcust_opapi.so"
export FLA_NPU_STABLE_LIB="$runtime_dir/libfla_npu_stable.so"
export ASCEND_CUSTOM_OPP_PATH="$vendor_dir:$full_runtime_vendor"
python - "$FLA_NPU_OP_API_LIB" <<'PY'
import ctypes
import sys

library = ctypes.CDLL(sys.argv[1])
for symbol in ("aclnnChunkGatedDeltaRuleFwd", "aclnnChunkGatedDeltaRuleFwdPool", "aclnnCausalConv1dGetWorkspaceSize"):
    getattr(library, symbol)
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd, npu_chunk_gated_delta_rule_fwd_pool

assert callable(chunk_gated_delta_rule_fwd)
assert callable(npu_chunk_gated_delta_rule_fwd_pool)
PY

sha256sum \
    "$runtime_dir/libfla_npu_stable.so" \
    "$op_api_dir/libcust_opapi.so" \
    "$pool_op_api" \
    "$full_op_api" \
    >"$output_dir/artifacts.sha256"
