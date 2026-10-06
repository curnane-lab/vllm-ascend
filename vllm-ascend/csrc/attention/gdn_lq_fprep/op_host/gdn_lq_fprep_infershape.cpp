#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4GdnLqFprep(gert::InferShapeContext* context)
{
    auto idxShape = context->GetInputShape(1);
    auto attrs = context->GetAttrs();
    if (idxShape == nullptr || idxShape->GetDimNum() < 1
        || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t t = idxShape->GetDim(0);
    // h from the attr: state arrives as a 1-element base view (the pool
    // base pointer; executor would otherwise materialize the giant
    // strided view), so its shape carries no head count
    int64_t h = *(attrs->GetAttrPointer<int64_t>(0));
    gert::Shape* abStage = context->GetOutputShape(0);
    gert::Shape* flags = context->GetOutputShape(1);
    gert::Shape* baStage = context->GetOutputShape(2);
    if (abStage == nullptr || flags == nullptr || baStage == nullptr) {
        return ge::GRAPH_FAILED;
    }
    flags->SetDimNum(2);
    flags->SetDim(0, t);
    flags->SetDim(1, h);
    abStage->SetDimNum(4);
    abStage->SetDim(0, t);
    abStage->SetDim(1, h);
    abStage->SetDim(2, 192);
    abStage->SetDim(3, 128);
    baStage->SetDimNum(4);
    baStage->SetDim(0, t);
    baStage->SetDim(1, h);
    baStage->SetDim(2, 192);
    baStage->SetDim(3, 128);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4GdnLqFprep(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GdnLqFprep)
    .InferShape(InferShape4GdnLqFprep)
    .InferDataType(InferDataType4GdnLqFprep);

}  // namespace ops
