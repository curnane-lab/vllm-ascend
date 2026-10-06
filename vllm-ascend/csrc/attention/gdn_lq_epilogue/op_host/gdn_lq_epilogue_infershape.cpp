#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqEpilogue(gert::InferShapeContext* context)
{
    auto vShape = context->GetInputShape(5);
    if (vShape == nullptr || vShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    // o mirrors v [T, h, V], bf16
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(3);
    outShape->SetDim(0, vShape->GetDim(0));
    outShape->SetDim(1, vShape->GetDim(1));
    outShape->SetDim(2, vShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqEpilogue(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqEpilogue)
    .InferShape(InferShape4GdnLqEpilogue)
    .InferDataType(InferDataType4GdnLqEpilogue);

}  // namespace ops
