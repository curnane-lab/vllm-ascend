#include "gdn_lq_epilogue_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc4GdnLqEpilogue(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    auto idxShape = context->GetInputShape(1);
    if (idxShape == nullptr || idxShape->GetStorageShape().GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batch = static_cast<uint32_t>(idxShape->GetStorageShape().GetDim(0));

    auto vShape = context->GetInputShape(5);
    if (vShape == nullptr || vShape->GetStorageShape().GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    uint32_t h = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(1));

    auto knShape = context->GetInputShape(3);
    if (knShape == nullptr || knShape->GetStorageShape().GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    uint32_t nk = static_cast<uint32_t>(knShape->GetStorageShape().GetDim(1));

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // attr order: 0 h, 1 nk, 2 s_slot (bytes), 3 scale
    uint32_t hAttr = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(0)));
    uint32_t nkAttr = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(1)));
    int64_t sSlot = *(attrs->GetAttrPointer<int64_t>(2));
    float scale = *(attrs->GetAttrPointer<float>(3));
    if (hAttr != h || nkAttr != nk || nk == 0 || h % nk != 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t numInstances = batch * h;
    GdnLqEpilogueTilingData tiling;
    tiling.set_numInstances(numInstances);
    uint32_t usedCoreNum = numInstances < coreNum ? numInstances : coreNum;
    tiling.set_instancesPerCore(numInstances / usedCoreNum);
    tiling.set_remainderCores(numInstances % usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_h(h);
    tiling.set_nk(nk);
    tiling.set_sSlot(sSlot);
    tiling.set_scale(scale);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnLqEpilogue)
    .Tiling(TilingFunc4GdnLqEpilogue);

}  // namespace optiling
