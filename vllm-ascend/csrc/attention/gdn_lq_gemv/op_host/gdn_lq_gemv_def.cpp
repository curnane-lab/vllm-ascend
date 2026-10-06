#include "register/op_def_registry.h"

namespace ops {
class GdnLqGemv : public OpDef {
public:
    explicit GdnLqGemv(const char* name) : OpDef(name)
    {
        // M2-D2 step v2, stage 2 (AIC-only): per (token, value-head) the
        // two codes row-dots on the cube:
        //   cq[t,h,v,0] = sum_k staging[t,h,v,k] * aq[t,h,0,k]   (codesKn)
        //   cq[t,h,v,1] = sum_k staging[t,h,v,k] * aq[t,h,1,k]   (codesQn)
        // staging = fp16(codes) [V,K]; aq = fp16 [16,K] with rows 0/1 =
        // k'/q' (s_k, 1/127 and K**-0.5 premultiplied by stage 1) and zero
        // pad rows. fp16 x fp16 -> fp32 accumulate (exact products).
        this->Input("staging").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("aq").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cq").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqGemv);

}  // namespace ops
