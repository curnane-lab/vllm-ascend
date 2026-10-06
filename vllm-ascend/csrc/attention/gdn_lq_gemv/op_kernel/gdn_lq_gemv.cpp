#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

// M2-D2 step v2, stage 2 (AIC-only). Per (token, value-head) instance, one
// cube GEMM computes the two v1 codes-row dots at once:
//   cq[inst, 0, v] = sum_k aq[inst, 0, k] * staging[inst, v, k]  (codesKn)
//   cq[inst, 1, v] = sum_k aq[inst, 1, k] * staging[inst, v, k]  (codesQn)
// aq is the A operand [M=16, K=128] (rows 0/1 = k'/q', pad rows zero);
// staging = fp16(codes) [V,K] is read as the transposed B operand (logical
// [K, V]). C rows 0/1 are contiguous [V] fp32 for the stage-3 epilogue.
// fp16 x fp16 -> fp32: products exact, fp32 accumulation.
// Stage 1 produced both operands; the epilogue (stage 3) consumes columns
// 0/1. Cross-kernel ordering is stream-implied (MIX unavailable, see
// results/M2-D1-probe.md).

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr uint32_t M_PAD = 16;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using MM = matmul::MatmulImpl<AType, BType, CType, BiasType>;

extern "C" __global__ __aicore__ void gdn_lq_gemv(
    GM_ADDR staging, GM_ADDR aq, GM_ADDR cq, GM_ADDR workspace, GM_ADDR tiling)
{
    ENABLE_FEATURE_FOR_COMPILE(default, KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    uint32_t core = GetBlockIdx();
    if (core >= tilingData.usedCoreNum) {
        return;
    }
    uint32_t myCount = tilingData.instancesPerCore;
    if (core < tilingData.remainderCores) {
        myCount += 1;
    }

    MM mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tilingData.mmTilingData, &pipe);

    __gm__ half* stageBase = reinterpret_cast<__gm__ half*>(staging);
    __gm__ half* aqBase = reinterpret_cast<__gm__ half*>(aq);
    __gm__ float* cqBase = reinterpret_cast<__gm__ float*>(cq);
    for (uint32_t m = 0; m < myCount; ++m) {
        uint32_t g = core + m * tilingData.usedCoreNum;
        GlobalTensor<half> gmA;
        gmA.SetGlobalBuffer(aqBase + static_cast<uint64_t>(g) * M_PAD * F_K);
        GlobalTensor<half> gmB;
        gmB.SetGlobalBuffer(stageBase + static_cast<uint64_t>(g) * F_V * F_K);
        GlobalTensor<float> gmC;
        gmC.SetGlobalBuffer(cqBase + static_cast<uint64_t>(g) * M_PAD * F_V);

        mm.SetTensorA(gmA);
        mm.SetTensorB(gmB, true);  // staging [V,K] row-major = B^T [K,V]
        mm.IterateAll(gmC);
    }
    mm.End();
}
