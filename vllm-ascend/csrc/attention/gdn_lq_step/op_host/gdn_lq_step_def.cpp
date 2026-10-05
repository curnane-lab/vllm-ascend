#include "register/op_def_registry.h"

namespace ops {
class GdnLqStep : public OpDef {
public:
    explicit GdnLqStep(const char* name) : OpDef(name)
    {
        // state is the fp32 [SLOTS, H, V, K] ssm_state slot buffer holding
        // the LeapQuant composite checkpoint + window buffer (same layout as
        // GdnLqFlush). Slots may be strided (unified KV pool pages):
        // IgnoreContiguous — the kernel addresses slots through the s_slot
        // byte-stride attribute.
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("idx").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // kn/qn: l2-normalized bf16 [T, H, K], expanded to value heads
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
        this->Output("out").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();
        // slot stride in BYTES (state.stride(0) * 4 of the strided view)
        this->Attr("s_slot").Int();
        this->Attr("scale").Float();

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqStep);

}  // namespace ops
