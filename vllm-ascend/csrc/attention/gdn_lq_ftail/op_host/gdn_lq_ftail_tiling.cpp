#include "gdn_lq_ftail_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc4GdnLqFtail(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    auto idxShape = context->GetInputShape(1);
    auto cShape = context->GetInputShape(3);
    if (idxShape == nullptr || cShape == nullptr
        || cShape->GetStorageShape().GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    uint32_t numInstances =
        static_cast<uint32_t>(cShape->GetStorageShape().GetDim(0))
        * static_cast<uint32_t>(cShape->GetStorageShape().GetDim(1));

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // attr order: 0 h, 1 s_slot (bytes), 2 phase
    uint32_t hAttr = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(0)));
    int64_t sSlot = *(attrs->GetAttrPointer<int64_t>(1));
    uint32_t phase = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(2)));
    if (phase < 1 || phase > 3) {
        return ge::GRAPH_FAILED;
    }

    GdnLqFtailTilingData tiling;
    tiling.set_numInstances(numInstances);
    uint32_t usedCoreNum = numInstances < coreNum ? numInstances : coreNum;
    tiling.set_instancesPerCore(numInstances / usedCoreNum);
    tiling.set_remainderCores(numInstances % usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_h(hAttr);
    tiling.set_sSlot(sSlot);
    tiling.set_phase(phase);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnLqFtail)
    .Tiling(TilingFunc4GdnLqFtail);

}  // namespace optiling
