#include "gdn_lq_fprep_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc4GdnLqFprep(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    auto idxShape = context->GetInputShape(1);
    auto attrs = context->GetAttrs();
    if (idxShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batch = static_cast<uint32_t>(idxShape->GetStorageShape().GetDim(0));
    // attr order: 0 h, 1 s_slot (bytes); state is a 1-element base view
    uint32_t h = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(0)));
    int64_t sSlot = *(attrs->GetAttrPointer<int64_t>(1));

    uint32_t numInstances = batch * h;
    GdnLqFprepTilingData tiling;
    tiling.set_numInstances(numInstances);
    uint32_t usedCoreNum = numInstances < coreNum ? numInstances : coreNum;
    tiling.set_instancesPerCore(numInstances / usedCoreNum);
    tiling.set_remainderCores(numInstances % usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_h(h);
    tiling.set_sSlot(sSlot);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnLqFprep)
    .Tiling(TilingFunc4GdnLqFprep);

}  // namespace optiling
