#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqFlush(gert::InferShapeContext* context)
{
    auto stateShape = context->GetInputShape(0);
    if (stateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // in-place op: output mirrors the state tensor
    *outShape = *stateShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqFlush(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqFlush)
    .InferShape(InferShape4GdnLqFlush)
    .InferDataType(InferDataType4GdnLqFlush);

}  // namespace ops
