#include "register/op_def_registry.h"

namespace ops {
class GdnLqFprep : public OpDef {
public:
    explicit GdnLqFprep(const char* name) : OpDef(name)
    {
        // M3-3b stage F1 (AIV): per due (slot, head), assemble the flush
        // staging on GM:
        //   s_stage[g]  fp32 [V,K] = p * codes * (s_k/127) x s_v   (D term)
        //   ab_stage[g] fp16 [32,V] = [q (4 rows); ubuf (16 rows); 12 zero]
        //                             (logical A = this^T [V,32])
        //   ba_stage[g] fp16 [32,K] = [u (4 rows); w*j kbuf (16 rows); zero]
        //                             (logical B [32,K])
        // g = b*h + h, instance mapping identical to gdn_lq_step. Due =
        // fill >= L (M3-1 semantics); non-due heads cost one 4 B fill read.
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("idx").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("ab_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("flags").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("ba_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();       // value heads per token
        this->Attr("s_slot").Int();  // slot stride in BYTES

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqFprep);

}  // namespace ops
