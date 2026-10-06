#include "gdn_lq_fgemv_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"

namespace optiling {

// per-phase cube geometry (M, N, K)
static void PhaseShape(uint32_t phase, int32_t& m, int32_t& n, int32_t& k)
{
    if (phase == 1) {
        m = 128;
        n = 128;
        k = 32;   // S = D + A^T B
    } else if (phase == 2 || phase == 4) {
        m = 128;
        n = 16;
        k = 128;  // m1 = S^T q0 / uk = S^T qv'
    } else {
        m = 128;
        n = 16;
        k = 128;  // uk = S^T qv'
    }
}

static ge::graphStatus TilingFunc4GdnLqFgemv(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();

    auto idxShape = context->GetInputShape(1);
    auto sShape = context->GetInputShape(2);
    if (idxShape == nullptr || sShape == nullptr
        || sShape->GetStorageShape().GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    uint32_t numInstances =
        static_cast<uint32_t>(sShape->GetStorageShape().GetDim(0))
        * static_cast<uint32_t>(sShape->GetStorageShape().GetDim(1));

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // attr order: 0 h, 1 s_slot (bytes), 2 phase
    int64_t sSlot = *(attrs->GetAttrPointer<int64_t>(1));
    uint32_t phase = static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(2)));
    if (phase < 1 || phase > 4) {
        return ge::GRAPH_FAILED;
    }

    GdnLqFgemvTilingData tiling;
    tiling.set_numInstances(numInstances);
    uint32_t usedCoreNum = numInstances < coreNum ? numInstances : coreNum;
    tiling.set_instancesPerCore(numInstances / usedCoreNum);
    tiling.set_remainderCores(numInstances % usedCoreNum);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_h(static_cast<uint32_t>(*(attrs->GetAttrPointer<int64_t>(0))));
    tiling.set_sSlot(sSlot);
    tiling.set_phase(phase);

    using namespace matmul_tiling;
    MatmulApiTiling mmTiling(ascendcPlatform);
    mmTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mmTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mmTiling.SetBias(false);
    mmTiling.SetShape(128, 128, 192);  // phase 1: S = A^T B (D folded in)
    mmTiling.SetOrgShape(128, 128, 192);
    mmTiling.SetBufferSpace(-1, -1, -1);
    mmTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    int r1 = mmTiling.GetTiling(tiling.mmTilingData);
    if (r1 == -1) {
        return ge::GRAPH_FAILED;
    }
    // a fresh tiling object: re-SetShape on a used one keeps stale state
    matmul_tiling::MatmulApiTiling mmTiling2(ascendcPlatform);
    mmTiling2.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mmTiling2.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    mmTiling2.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    mmTiling2.SetBias(false);
    mmTiling2.SetShape(128, 32, 128);  // phases 2-4 projections, fp32 C
    mmTiling2.SetOrgShape(128, 32, 128);
    mmTiling2.SetBufferSpace(-1, -1, -1);
    if (mmTiling2.GetTiling(tiling.mmTilingData2) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnLqFgemv)
    .Tiling(TilingFunc4GdnLqFgemv);

}  // namespace optiling
