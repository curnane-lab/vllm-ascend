#include "register/op_def_registry.h"

namespace ops {
class GdnLqEpilogue : public OpDef {
public:
    explicit GdnLqEpilogue(const char* name) : OpDef(name)
    {
        // M2-D2 step v2, stage 3 (AIV-only): the v1 GdnLqStep assemble half
        // with the two codes row-dots (256 WholeReduceSum per head in v1)
        // replaced by the stage-2 cube output cq rows 0/1 read from GM.
        // Per (slot, value-head): full decode-step contract - compensator
        // token terms, live-window updates, w decay, p accumulate, in-place
        // window append - plus o = Sqn + kq*u_new. kn/qn arrive unexpanded
        // per key head (value head h uses key head h//rep), same as stage 1.
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("idx").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // stage-2 cube output [T, h, 16, V]; rows 0/1 = codesKn/codesQn
        this->Input("cq").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kn").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("qn").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("g_log").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();       // value heads per token
        this->Attr("nk").Int();      // key heads per token; rep = h / nk
        this->Attr("s_slot").Int();  // slot stride in BYTES
        this->Attr("scale").Float(); // K**-0.5 applied to qn

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqEpilogue);

}  // namespace ops
