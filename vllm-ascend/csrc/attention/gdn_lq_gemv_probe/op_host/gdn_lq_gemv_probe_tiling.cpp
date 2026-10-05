#include "gdn_lq_gemv_probe_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"

namespace optiling {

// fixed cube geometry per instance: M=16 (q rows padded from n<=8),
// K=128, N=V=128, single tile, no K/M/N loop
static constexpr int32_t MM_M = 16;
static constexpr int32_t MM_N = 128;
static constexpr int32_t MM_K = 128;

static ge::graphStatus TilingFunc4GdnLqGemvProbe(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    // AIC-only kernel: blockDim counts AIC cores
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();

    auto aShape = context->GetInputShape(0);
    if (aShape == nullptr || aShape->GetStorageShape().GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    uint32_t g = static_cast<uint32_t>(aShape->GetStorageShape().GetDim(0));
    if (aShape->GetStorageShape().GetDim(1) != MM_M
        || aShape->GetStorageShape().GetDim(2) != MM_K) {
        return ge::GRAPH_FAILED;
    }

    GdnLqGemvProbeTilingData tiling;
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

IMPL_OP_OPTILING(GdnLqGemvProbe)
    .Tiling(TilingFunc4GdnLqGemvProbe);

}  // namespace optiling
