// Stable-ABI adapter for the Phase6 indexed pool forward.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/layout_math.h"

#include <cstdint>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::tensor;

constexpr const char* kSchema_chunk_gated_delta_rule_fwd_pool =
    "npu_chunk_gated_delta_rule_fwd_pool(Tensor q, Tensor k, Tensor v, Tensor g, "
    "Tensor beta, Tensor pool_state, Tensor source_slots, Tensor destination_slots, "
    "Tensor has_initial, Tensor? cu_seqlens, Tensor? chunk_indices, float scale, "
    "int chunk_size, int stream) -> Tensor";

Tensor run_npu_chunk_gated_delta_rule_fwd_pool(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor pool_state,
    Tensor source_slots, Tensor destination_slots, Tensor has_initial,
    std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices,
    double scale, int64_t chunk_size, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta pool_meta = meta_of(pool_state);
  const int64_t batch = q_meta.sizes[0];
  const int64_t tokens = q_meta.sizes[2];
  const int64_t value_heads = v_meta.sizes[1];
  Tensor output = allocate_sizes({batch, tokens, value_heads, v_meta.sizes[3]},
                                 v_meta.scalar_type, v_meta);
  FLA_STABLE_EXEC(
      "aclnnChunkGatedDeltaRuleFwdPool", q_meta, stream,
      tensor(q_meta), tensor(k_meta), tensor(v_meta), tensor(meta_of(g)),
      tensor(meta_of(beta)), tensor(pool_meta), int_array(cu_seqlens),
      int_array(chunk_indices), tensor(meta_of(source_slots)),
      tensor(meta_of(destination_slots)), tensor(meta_of(has_initial)),
      scalar(scale), scalar(chunk_size), out_tensor(meta_of(output)),
      out_tensor(pool_meta));
  return output;
}

} // namespace
