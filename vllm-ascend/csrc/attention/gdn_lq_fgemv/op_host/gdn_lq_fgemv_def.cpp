#include "register/op_def_registry.h"

namespace ops {
class GdnLqFgemv : public OpDef {
public:
    explicit GdnLqFgemv(const char* name) : OpDef(name)
    {
        // M3-3b cube stage (AIC), attr phase:
        //   phase 1: S = D + A^T B (atomic accumulate onto s_stage), A =
        //            ab_stage^T [V,32], B = ba_stage [32,K]
        //   phase 2: m1 = S^T q0 -> c_stage [K,16] (cols 0..3 used)
        //   phase 3: qv = S qk -> c_stage [V,16] (B = kq_stage^T)
        //   phase 4: uk = S^T qv' -> c_stage [K,16] (B = vq_stage^T)
        // Due heads only (fill >= L); non-due cost one 4 B fill read.
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("idx").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("s_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ab_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("flags").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ba_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // shared padded quantizer queries [V,16] fp16 (cols 0..3 = q0)
        this->Input("q0_pad").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kq_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("vq_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();
        this->Attr("s_slot").Int();
        this->Attr("phase").Int();  // 1, 2 or 3

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqFgemv);

}  // namespace ops
