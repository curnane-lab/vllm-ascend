#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

// M3-3b cube stage (AIC). attr phase:
//   1: S(fp16) = A^T B  (D folded into the GEMM via the diagonal trick;
//      plain overwrite, no atomic)   A = ab_stage [192,V], B = ba_stage
//   2: m1 = S^T q0 -> c_stage [K,16] fp32
//   3: qv = S qk    -> c_stage [V,16]
//   4: uk = S^T qv' -> c_stage [K,16]
// Due heads only (fill >= L); non-due heads cost one 4 B fill read via
// scalar GM GetValue (AIC has no UB staging).

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr uint32_t F_PAD = 192;
constexpr uint32_t OFF_FILL = 27716;
constexpr uint32_t HEAD_BYTES = 32768;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using CS16 = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using CS32 = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using MM16 = matmul::MatmulImpl<AType, BType, CS16, BiasType>;
using MM32 = matmul::MatmulImpl<AType, BType, CS32, BiasType>;

// NOTE: AIC scalar GM reads (GlobalTensor::GetValue) fault on this stack
// (verified empirically), so fgemv processes EVERY instance; the staging
// of non-due heads is written but never consumed (ftail re-checks due on
// the AIV side before reading staging or writing state). The host wrapper
// short-circuits the remaining launches when fprep's due count is zero.

extern "C" __global__ __aicore__ void gdn_lq_fgemv(
    GM_ADDR state, GM_ADDR idx, GM_ADDR s_stage, GM_ADDR ab_stage,
    GM_ADDR flags, GM_ADDR ba_stage, GM_ADDR q0_pad, GM_ADDR kq_stage,
    GM_ADDR vq_stage, GM_ADDR c_stage, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)state; (void)idx; (void)flags;
    ENABLE_FEATURE_FOR_COMPILE(default, KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    uint32_t core = GetBlockIdx();
    if (core >= tilingData.usedCoreNum) {
        return;
    }
    uint32_t myCount = tilingData.instancesPerCore;
    uint32_t myStart;
    if (core < tilingData.remainderCores) {
        myStart = core * (tilingData.instancesPerCore + 1);
        myCount += 1;
    } else {
        myStart = tilingData.remainderCores * (tilingData.instancesPerCore + 1)
            + (core - tilingData.remainderCores) * tilingData.instancesPerCore;
    }
    uint32_t phase = tilingData.phase;

    __gm__ half* fgemvS = reinterpret_cast<__gm__ half*>(s_stage);
    __gm__ half* fgemvAB = reinterpret_cast<__gm__ half*>(ab_stage);
    __gm__ half* fgemvBA = reinterpret_cast<__gm__ half*>(ba_stage);
    __gm__ half* fgemvQ0 = reinterpret_cast<__gm__ half*>(q0_pad);
    __gm__ half* fgemvKQ = reinterpret_cast<__gm__ half*>(kq_stage);
    __gm__ half* fgemvVQ = reinterpret_cast<__gm__ half*>(vq_stage);
    __gm__ float* fgemvC = reinterpret_cast<__gm__ float*>(c_stage);

    if (phase == 1) {
        MM16 mm;
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.mmTilingData, &pipe);
        for (uint32_t i = 0; i < myCount; ++i) {
            uint32_t inst = myStart + i;
            uint64_t g = inst;
            GlobalTensor<half> gmAB;
            gmAB.SetGlobalBuffer(fgemvAB + g * F_PAD * F_V);
            GlobalTensor<half> gmBA;
            gmBA.SetGlobalBuffer(fgemvBA + g * F_PAD * F_K);
            GlobalTensor<half> gmS;
            gmS.SetGlobalBuffer(fgemvS + g * F_V * F_K);
            mm.SetTensorA(gmAB, true);  // logical A [V,192]
            mm.SetTensorB(gmBA);        // B [192,K]
            mm.IterateAll(gmS);
        }
        mm.End();
    } else {
        MM32 mm;
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.mmTilingData2, &pipe);
        for (uint32_t i = 0; i < myCount; ++i) {
            uint32_t inst = myStart + i;
            uint64_t g = inst;
            GlobalTensor<half> gmS;
            gmS.SetGlobalBuffer(fgemvS + g * F_V * F_K);
            GlobalTensor<float> gmC;
            gmC.SetGlobalBuffer(fgemvC + g * F_V * 32);
            if (phase == 2) {
                GlobalTensor<half> gmQ0;
                gmQ0.SetGlobalBuffer(fgemvQ0);
                mm.SetTensorA(gmS, true);   // logical S^T [K,V]
                mm.SetTensorB(gmQ0);        // q0 padded [V,16]
                mm.IterateAll(gmC);
            } else if (phase == 3) {
                GlobalTensor<half> gmKQ;
                gmKQ.SetGlobalBuffer(fgemvKQ + g * F_K * 32);
                mm.SetTensorA(gmS);
                mm.SetTensorB(gmKQ);        // kq pre-transposed [K,32]
                mm.IterateAll(gmC);
            } else {
                GlobalTensor<half> gmVQ;
                gmVQ.SetGlobalBuffer(fgemvVQ + g * F_V * 32);
                mm.SetTensorA(gmS, true);
                mm.SetTensorB(gmVQ);        // vq pre-transposed [V,32]
                mm.IterateAll(gmC);
            }
        }
        mm.End();
    }
}
