#include "gdn_sketch_probe_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc4GdnSketchProbe(gert::TilingContext* context)
{
    uint32_t coreNum;
    auto ptrCompileInfo = reinterpret_cast<const GdnSketchProbeCompileInfo*>(
        context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());
        coreNum = ascendcPlatform.GetCoreNum();
    } else {
        coreNum = ptrCompileInfo->totalCoreNum;
    }

    auto qShape = context->GetInputShape(1);
    if (qShape == nullptr || qShape->GetStorageShape().GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    uint32_t numInstances =
        static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));

    int32_t mode = 99;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        mode = *(attrs->GetAttrPointer<int32_t>(0));
    }

    GdnSketchProbeTilingData tiling;
    tiling.set_numInstances(numInstances);
    tiling.set_mode(static_cast<uint32_t>(mode));
    uint32_t usedCoreNum = numInstances < coreNum ? numInstances : coreNum;
    uint32_t perCore = numInstances / usedCoreNum;
    uint32_t remainder = numInstances % usedCoreNum;
    tiling.set_instancesPerCore(perCore);
    tiling.set_remainderCores(remainder);
    tiling.set_usedCoreNum(usedCoreNum);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GdnSketchProbe(
    gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GdnSketchProbeCompileInfo>();
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

IMPL_OP_OPTILING(GdnSketchProbe)
    .Tiling(TilingFunc4GdnSketchProbe)
    .TilingParse<GdnSketchProbeCompileInfo>(TilingPrepare4GdnSketchProbe);

}  // namespace optiling
