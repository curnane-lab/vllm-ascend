#include "gdn_lq_step_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc4GdnLqStep(gert::TilingContext* context)
{
    uint32_t coreNum;
    auto ptrCompileInfo = reinterpret_cast<const GdnLqStepCompileInfo*>(
        context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());
        // AIV-only kernel (KERNEL_TYPE_AIV_ONLY): schedule on all vector
        // cores, not the (possibly smaller) AI-core count
        coreNum = ascendcPlatform.GetCoreNumAiv();
    } else {
        coreNum = ptrCompileInfo->totalCoreNum;
    }

    auto idxShape = context->GetInputShape(1);
    if (idxShape == nullptr || idxShape->GetStorageShape().GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batch = static_cast<uint32_t>(idxShape->GetStorageShape().GetDim(0));

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // attr order: 0 h, 1 s_slot (bytes), 2 scale
    uint32_t h = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(0)));
    int64_t sSlot = *(attrs->GetAttrPointer<int64_t>(1));
    float scale = *(attrs->GetAttrPointer<float>(2));
    uint32_t numInstances = batch * h;

    GdnLqStepTilingData tiling;
    tiling.set_numInstances(numInstances);
    uint32_t usedCoreNum = numInstances < coreNum ? numInstances : coreNum;
    tiling.set_instancesPerCore(numInstances / usedCoreNum);
    tiling.set_remainderCores(numInstances % usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_batch(batch);
    tiling.set_h(h);
    tiling.set_sSlot(sSlot);
    tiling.set_scale(scale);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GdnLqStep(
    gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GdnLqStepCompileInfo>();
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnLqStep)
    .Tiling(TilingFunc4GdnLqStep)
    .TilingParse<GdnLqStepCompileInfo>(TilingPrepare4GdnLqStep);

}  // namespace optiling
