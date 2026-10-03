#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnSketchProbe(gert::InferShapeContext* context)
{
    auto qShape = context->GetInputShape(1);
    if (qShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(2);
    outShape->SetDim(0, qShape->GetDim(0));
    outShape->SetDim(1, 128);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnSketchProbe(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnSketchProbe)
    .InferShape(InferShape4GdnSketchProbe)
    .InferDataType(InferDataType4GdnSketchProbe);

}  // namespace ops
