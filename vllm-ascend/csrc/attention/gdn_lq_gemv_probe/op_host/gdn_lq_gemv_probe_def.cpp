#include "register/op_def_registry.h"

namespace ops {
class GdnLqGemvProbe : public OpDef {
public:
    explicit GdnLqGemvProbe(const char* name) : OpDef(name)
    {
        // M2-D1 probe: LeapQuant decode-step core GEMV on the cube core.
        //   o[g,n,v] = sum_k (codes[g,k,v] * scale[g,k]) * q[g,n,k]
        // factorized as a cube fp16 GEMM per instance g:
        //   C[m,v] = sum_k A[m,k] * B[k,v]
        //   A = fp16(fp32(q[g,n,:]) * scale[g,:]) zero-padded to [16,128]
        //   B = fp16(codes[g,:,:])                  exact  [128,128]
        //   C = fp32 [16,128] (rows [n,16) are the zero-pad products)
        // On this toolchain the AIV->AIC MIX launch path (KERNEL_TYPE_MIX_*,
        // fftsplus task) faults on an EMPTY kernel (see results/M2-D1-probe.md),
        // so the probe is an AIC-ONLY kernel: A/B tiles are produced by the
        // vector path (verified on-chip dequant in this file's history / the
        // probe report) or by torch, and the cube GEMV runs stream-ordered.
        this->Input("a").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GdnLqGemvProbe);

}  // namespace ops
