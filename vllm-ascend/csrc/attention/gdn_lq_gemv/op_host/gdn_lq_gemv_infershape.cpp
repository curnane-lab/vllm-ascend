#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqGemv(gert::InferShapeContext* context)
{
    auto stagingShape = context->GetInputShape(0);
    if (stagingShape == nullptr || stagingShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    // cq [T, h, 16, V=128] fp32 (rows 0/1 = codesKn/codesQn)
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(4);
    outShape->SetDim(0, stagingShape->GetDim(0));
    outShape->SetDim(1, stagingShape->GetDim(1));
    outShape->SetDim(2, 16);
    outShape->SetDim(3, stagingShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqGemv(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqGemv)
    .InferShape(InferShape4GdnLqGemv)
    .InferDataType(InferDataType4GdnLqGemv);

}  // namespace ops
