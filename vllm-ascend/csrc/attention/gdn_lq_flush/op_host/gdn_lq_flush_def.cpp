#include "register/op_def_registry.h"

namespace ops {
class GdnLqFlush : public OpDef {
public:
    explicit GdnLqFlush(const char* name) : OpDef(name)
    {
        // state is the fp32 [SLOTS, H, V, K] ssm_state slot buffer; the
        // LeapQuant composite checkpoint lives in the first LAYOUT_END bytes
        // of each slot. The op updates it in place: the caller passes the
        // same tensor as the output. Slots may be strided (unified KV pool
        // pages): IgnoreContiguous — the kernel addresses slots through the
        // s_slot byte-stride attribute. AutoContiguous would stage a hidden
        // contiguous copy (breaking the in-place aliasing and the slot
        // layout); IgnoreContiguous declares the op handles strides itself.
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("idx").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // q0 pre-transposed to [R, V] fp32 (row r is the torch q0[:, r])
        this->Input("q0").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();
        // slot stride in BYTES (state.stride(0) * 4 of the strided view)
        this->Attr("s_slot").Int();

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqFlush);

}  // namespace ops
