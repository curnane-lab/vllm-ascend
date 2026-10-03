#include "register/op_def_registry.h"

namespace ops {
class GdnSketchStep : public OpDef {
public:
    explicit GdnSketchStep(const char* name) : OpDef(name)
    {
        // inputs follow the triton launcher argument order
        this->Input("mixed_qkv").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("a_act").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("b_act").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("a_log").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dt_bias").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("d_cache").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("k_cache").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("g_cache").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("slots").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("write_pos").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("meta").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("u").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("phi").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("fs").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("beta_ring").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("current_d").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("current_k").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("ranks").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("layout").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("out").ParamType(REQUIRED)
            .DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("scale").Float();
        this->Attr("batch").Int();
        this->Attr("hv").Int();
        this->Attr("h").Int();
        this->Attr("s_qkv").Int();
        this->Attr("s_a").Int();
        this->Attr("s_b").Int();
        this->Attr("s_st_slot").Int();
        this->Attr("s_st_head").Int();
        this->Attr("s_dc_slot").Int();
        this->Attr("s_dc_head").Int();
        this->Attr("s_kc_slot").Int();
        this->Attr("s_kc_head").Int();
        this->Attr("s_gc_slot").Int();
        this->Attr("s_gc_head").Int();
        this->Attr("s_su").Int();
        this->Attr("s_sm").Int();
        this->Attr("s_sf").Int();

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnSketchStep);

}  // namespace ops
