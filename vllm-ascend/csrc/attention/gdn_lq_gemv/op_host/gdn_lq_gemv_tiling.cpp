#include "gdn_lq_gemv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"

namespace optiling {

// fixed cube geometry per instance: A = aq [M=16, K=128] direct (rows 0/1 =
// k'/q', pad rows zero), B = staging read transposed (logical [K=128,
// N=V=128] from physical [V,K]), C = cq [16, 128] fp32 (rows 0/1 are the
// two dots, contiguous for the epilogue)
static constexpr int32_t MM_M = 16;
static constexpr int32_t MM_N = 128;
static constexpr int32_t MM_K = 128;

static ge::graphStatus TilingFunc4GdnLqGemv(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();

    auto stagingShape = context->GetInputShape(0);
    if (stagingShape == nullptr
        || stagingShape->GetStorageShape().GetDimNum() != 4
        || stagingShape->GetStorageShape().GetDim(2) != MM_N
        || stagingShape->GetStorageShape().GetDim(3) != MM_K) {
        return ge::GRAPH_FAILED;
    }
    uint32_t g = static_cast<uint32_t>(stagingShape->GetStorageShape().GetDim(0))
        * static_cast<uint32_t>(stagingShape->GetStorageShape().GetDim(1));

    GdnLqGemvTilingData tiling;
    tiling.set_numInstances(g);
    uint32_t usedCoreNum = g < coreNum ? g : coreNum;
    tiling.set_instancesPerCore(g / usedCoreNum);
    tiling.set_remainderCores(g % usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);

    using namespace matmul_tiling;
    MatmulApiTiling mmTiling(ascendcPlatform);
    mmTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mmTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mmTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    mmTiling.SetBias(false);
    mmTiling.SetShape(MM_M, MM_N, MM_K);
    mmTiling.SetOrgShape(MM_M, MM_N, MM_K);
    mmTiling.SetBufferSpace(-1, -1, -1);
    if (mmTiling.GetTiling(tiling.mmTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnLqGemv)
    .Tiling(TilingFunc4GdnLqGemv);

}  // namespace optiling
