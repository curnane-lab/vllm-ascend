#include "register/op_def_registry.h"

namespace ops {
class GdnLqFtail : public OpDef {
public:
    explicit GdnLqFtail(const char* name) : OpDef(name)
    {
        // M3-3b AIV tail, attr phase:
        //   1: qk  = mgs(c_stage cols 0..3) -> kq_stage [16,K] fp16
        //   2: qv' = mgs(c_stage cols 0..3) -> vq_stage [16,V] fp16
        //   3: quantize: E = S - qv' uk^T from s_stage/c_stage/vq_stage,
        //      sk/sv/codes, write back the composite checkpoint + w=0,
        //      p=1, fill=L (M3-1 semantics)
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("idx").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("s_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("vq_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("kq_stage").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();
        this->Attr("s_slot").Int();
        this->Attr("phase").Int();  // 1, 2 or 3

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqFtail);

}  // namespace ops
