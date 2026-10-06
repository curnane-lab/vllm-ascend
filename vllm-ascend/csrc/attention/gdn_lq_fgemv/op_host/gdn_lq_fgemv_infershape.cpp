#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqFgemv(gert::InferShapeContext* context)
{
    auto idxShape = context->GetInputShape(1);
    auto sShape = context->GetInputShape(2);
    if (idxShape == nullptr || sShape == nullptr
        || sShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* cShape = context->GetOutputShape(0);
    if (cShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    cShape->SetDimNum(4);
    cShape->SetDim(0, sShape->GetDim(0));
    cShape->SetDim(1, sShape->GetDim(1));
    cShape->SetDim(2, 128);
    cShape->SetDim(3, 32);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqFgemv(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqFgemv)
    .InferShape(InferShape4GdnLqFgemv)
    .InferDataType(InferDataType4GdnLqFgemv);

}  // namespace ops
