/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_pool_tiling.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <register/op_impl_registry.h>

namespace optiling {

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdPoolArch22(gert::TilingContext *context);

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdPool(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleFwdPool", "Invalid tiling context."),
                return ge::GRAPH_FAILED);
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    OP_CHECK_IF(platform.GetCurNpuArch() == NpuArch::DAV_3510,
                OP_LOGE("ChunkGatedDeltaRuleFwdPool", "Ascend950 pool I/O is not enabled."),
                return ge::GRAPH_FAILED);
    return Tiling4ChunkGatedDeltaRuleFwdPoolArch22(context);
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleFwdPool(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwdPool)
    .Tiling(Tiling4ChunkGatedDeltaRuleFwdPool)
    .TilingParse<ChunkGatedDeltaRuleFwdPoolCompileInfo>(TilingPrepareForChunkGatedDeltaRuleFwdPool);

} // namespace optiling
