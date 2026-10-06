#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqDequant(gert::InferShapeContext* context)
{
    auto idxShape = context->GetInputShape(1);
    auto knShape = context->GetInputShape(2);
    auto attrs = context->GetAttrs();
    if (idxShape == nullptr || knShape == nullptr || attrs == nullptr
        || knShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    int64_t t = idxShape->GetDim(0);
    int64_t h = *(attrs->GetAttrPointer<int64_t>(0));  // attr 0: h
    int64_t k = knShape->GetDim(2);
    gert::Shape* stagingShape = context->GetOutputShape(0);
    gert::Shape* aqShape = context->GetOutputShape(1);
    if (stagingShape == nullptr || aqShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // staging [T, h, V=128, K=128]
    stagingShape->SetDimNum(4);
    stagingShape->SetDim(0, t);
    stagingShape->SetDim(1, h);
    stagingShape->SetDim(2, 128);
    stagingShape->SetDim(3, k);
    // aq [T, h, 16, K] (cube M pad 16; rows 0/1 = k'/q', rest zero)
    aqShape->SetDimNum(4);
    aqShape->SetDim(0, t);
    aqShape->SetDim(1, h);
    aqShape->SetDim(2, 16);
    aqShape->SetDim(3, k);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqDequant(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqDequant)
    .InferShape(InferShape4GdnLqDequant)
    .InferDataType(InferDataType4GdnLqDequant);

}  // namespace ops
