#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqFtail(gert::InferShapeContext* context)
{
    auto idxShape = context->GetInputShape(1);
    auto cShape = context->GetInputShape(3);
    if (idxShape == nullptr || cShape == nullptr || cShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* vqShape = context->GetOutputShape(0);
    gert::Shape* kqShape = context->GetOutputShape(1);
    if (kqShape == nullptr || vqShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (gert::Shape* sh : {vqShape, kqShape}) {
        sh->SetDimNum(4);
        sh->SetDim(0, cShape->GetDim(0));
        sh->SetDim(1, cShape->GetDim(1));
        sh->SetDim(2, 128);
        sh->SetDim(3, 32);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqFtail(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqFtail)
    .InferShape(InferShape4GdnLqFtail)
    .InferDataType(InferDataType4GdnLqFtail);

}  // namespace ops
