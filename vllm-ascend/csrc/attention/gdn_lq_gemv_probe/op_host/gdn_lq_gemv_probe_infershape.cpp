#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqGemvProbe(gert::InferShapeContext* context)
{
    auto aShape = context->GetInputShape(0);
    if (aShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // c mirrors the A tile: [G, 16, 128], fp32
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(3);
    outShape->SetDim(0, aShape->GetDim(0));
    outShape->SetDim(1, aShape->GetDim(1));
    outShape->SetDim(2, aShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqGemvProbe(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqGemvProbe)
    .InferShape(InferShape4GdnLqGemvProbe)
    .InferDataType(InferDataType4GdnLqGemvProbe);

}  // namespace ops
