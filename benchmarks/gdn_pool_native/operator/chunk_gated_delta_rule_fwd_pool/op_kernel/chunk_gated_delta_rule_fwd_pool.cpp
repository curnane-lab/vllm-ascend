/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef GDN_FWD_POOL_IO
#define GDN_FWD_POOL_IO
#endif
#include "../original_phase6/op_kernel/internal/arch22/chunk_gated_delta_rule_fwd_arch22.cpp"

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_pool(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR a_storage, GM_ADDR raw_g,
    GM_ADDR gk, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR source_slots, GM_ADDR destination_slots, GM_ADDR has_initial,
    GM_ADDR o, GM_ADDR final_state, GM_ADDR g_cumsum_bth, GM_ADDR A,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)a_storage;
    REGISTER_TILING_DEFAULT(GDN::Arch22ChunkGatedDeltaRuleFwdTrailer);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RunPhase6<DTYPE_Q, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            q, k, v, beta, raw_g, gk, initial_state, cu_seqlens, chunk_indices,
            o, final_state, g_cumsum_bth, A, workspace, tiling,
            source_slots, destination_slots, has_initial);
    }
}
