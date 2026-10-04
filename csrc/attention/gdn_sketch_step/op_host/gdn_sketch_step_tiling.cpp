#include "gdn_sketch_step_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc4GdnSketchStep(gert::TilingContext* context)
{
    uint32_t coreNum;
    auto ptrCompileInfo = reinterpret_cast<const GdnSketchStepCompileInfo*>(
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

    auto qkvShape = context->GetInputShape(0);
    if (qkvShape == nullptr || qkvShape->GetStorageShape().GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batch = static_cast<uint32_t>(qkvShape->GetStorageShape().GetDim(0));

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float scale = *(attrs->GetAttrPointer<float>(0));
    // attr order: 0 scale, 1 batch, 2 hv, 3 h, 4..18 strides..., 18 mode
    uint32_t hv = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(2)));
    uint32_t h = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(3)));
    int64_t sQkv = *(attrs->GetAttrPointer<int64_t>(4));
    int64_t sA = *(attrs->GetAttrPointer<int64_t>(5));
    int64_t sB = *(attrs->GetAttrPointer<int64_t>(6));
    uint32_t numInstances = batch * hv;

    GdnSketchStepTilingData tiling;
    tiling.set_numInstances(numInstances);
    uint32_t usedCoreNum = numInstances < coreNum ? numInstances : coreNum;
    tiling.set_instancesPerCore(numInstances / usedCoreNum);
    tiling.set_remainderCores(numInstances % usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_batch(batch);
    tiling.set_hv(hv);
    tiling.set_h(h);
    tiling.set_scale(scale);
    tiling.set_sQkv(sQkv);
    tiling.set_sA(sA);
    tiling.set_sB(sB);
    tiling.set_sStSlot(*(attrs->GetAttrPointer<int64_t>(7)));
    tiling.set_sStHead(*(attrs->GetAttrPointer<int64_t>(8)));
    tiling.set_sDcSlot(*(attrs->GetAttrPointer<int64_t>(9)));
    tiling.set_sDcHead(*(attrs->GetAttrPointer<int64_t>(10)));
    tiling.set_sKcSlot(*(attrs->GetAttrPointer<int64_t>(11)));
    tiling.set_sKcHead(*(attrs->GetAttrPointer<int64_t>(12)));
    tiling.set_sGcSlot(*(attrs->GetAttrPointer<int64_t>(13)));
    tiling.set_sGcHead(*(attrs->GetAttrPointer<int64_t>(14)));
    tiling.set_sSu(*(attrs->GetAttrPointer<int64_t>(15)));
    tiling.set_sSm(*(attrs->GetAttrPointer<int64_t>(16)));
    tiling.set_sSf(*(attrs->GetAttrPointer<int64_t>(17)));
    tiling.set_mode(static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(18))));

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GdnSketchStep(
    gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GdnSketchStepCompileInfo>();
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

IMPL_OP_OPTILING(GdnSketchStep)
    .Tiling(TilingFunc4GdnSketchStep)
    .TilingParse<GdnSketchStepCompileInfo>(TilingPrepare4GdnSketchStep);

}  // namespace optiling
