#include "register/op_def_registry.h"

namespace ops {
class GdnLqDequant : public OpDef {
public:
    explicit GdnLqDequant(const char* name) : OpDef(name)
    {
        // M2-D2 step v2, stage 1 (AIV-only): per (slot, value-head) instance
        //   staging[t,h] = fp16(codes[slot, h])            [V,K] exact cast
        //   aq[t,h,0,:]  = fp16(kn[t, h//rep] * s_k / 127)
        //   aq[t,h,1,:]  = fp16(qn[t, h//rep] * scale * s_k / 127)
        //   aq[t,h,2:,:] = 0        (cube M-tail pad rows)
        // codes and s_k live in the composite per-head block of the fp32
        // ssm_state slot buffer (same layout as GdnLqStep); slots may be
        // strided (unified KV pool pages) via the s_slot byte-stride attr.
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("idx").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // per-key-head rows, NOT expanded to value heads (h//rep in kernel)
        this->Input("kn").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("qn").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("staging").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("aq").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();       // value heads per token (= state heads)
        this->Attr("nk").Int();      // key heads per token; rep = h / nk
        this->Attr("s_slot").Int();  // slot stride in BYTES
        this->Attr("scale").Float(); // K**-0.5 folded into the q' row

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqDequant);

}  // namespace ops
