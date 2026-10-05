#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqStep(gert::InferShapeContext* context)
{
    auto stateShape = context->GetInputShape(0);
    auto vShape = context->GetInputShape(4);
    if (stateShape == nullptr || vShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // out = [T, H, V]: T from v dim 0, H*V from v dims 1*2
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

static ge::graphStatus InferDataType4GdnLqStep(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqStep)
    .InferShape(InferShape4GdnLqStep)
    .InferDataType(InferDataType4GdnLqStep);

}  // namespace ops
