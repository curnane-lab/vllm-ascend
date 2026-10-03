#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnSketchStep(gert::InferShapeContext* context)
{
    auto qkvShape = context->GetInputShape(0);
    auto stateShape = context->GetInputShape(5);
    if (qkvShape == nullptr || stateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // out = (batch, HV, V): HV*V from state dims 1*2, batch from qkv dim 0
    outShape->SetDimNum(3);
    outShape->SetDim(0, qkvShape->GetDim(0));
    outShape->SetDim(1, stateShape->GetDim(1));
    outShape->SetDim(2, stateShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnSketchStep(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnSketchStep)
    .InferShape(InferShape4GdnSketchStep)
    .InferDataType(InferDataType4GdnSketchStep);

}  // namespace ops
