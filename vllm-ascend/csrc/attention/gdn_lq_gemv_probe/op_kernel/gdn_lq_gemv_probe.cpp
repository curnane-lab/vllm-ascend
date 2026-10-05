#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

// M2-D1 probe: LeapQuant decode-step core GEMV on the cube core.
//
//   o[g,n,v] = sum_k (codes[g,k,v] * scale[g,k]) * q[g,n,k]
//
// factorized so the cube sees pure fp16 operands (see op_host def): one
// 16x128x128 fp16 GEMM per instance g, C fp32 [16,128]:
//
//   C[g,m,v] = sum_k A[g,m,k] * B[g,k,v]
//
// Data path: the fp16 A/B tiles are produced off-cube (vector cores or
// torch), staged in GM (L2-resident on read), and the cube's MTE2 pulls
// them per instance; IterateAll writes C back to GM. On this toolchain
// the AIV->AIC in-kernel handshake (MIX cross-core flags) is unavailable:
// KERNEL_TYPE_MIX_* launches as an fftsplus task that faults on an EMPTY
// kernel on this device, and dav_l300 stubs CrossCoreSetFlag/WaitFlag to
// no-ops — so cross-core ordering is done with stream-ordered kernels.

constexpr uint32_t M_PAD = 16;
constexpr uint32_t K_DIM = 128;
constexpr uint32_t V_DIM = 128;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using MM = matmul::MatmulImpl<AType, BType, CType, BiasType>;

extern "C" __global__ __aicore__ void gdn_lq_gemv_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
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

    __gm__ half* aBase = reinterpret_cast<__gm__ half*>(a);
    __gm__ half* bBase = reinterpret_cast<__gm__ half*>(b);
    __gm__ float* cBase = reinterpret_cast<__gm__ float*>(c);
    for (uint32_t m = 0; m < myCount; ++m) {
        uint32_t g = core + m * tilingData.usedCoreNum;
        GlobalTensor<half> gmA;
        gmA.SetGlobalBuffer(aBase + static_cast<uint64_t>(g) * M_PAD * K_DIM);
        GlobalTensor<half> gmB;
        gmB.SetGlobalBuffer(bBase + static_cast<uint64_t>(g) * K_DIM * V_DIM);
        GlobalTensor<float> gmC;
        gmC.SetGlobalBuffer(cBase + static_cast<uint64_t>(g) * M_PAD * V_DIM);

        mm.SetTensorA(gmA);
        mm.SetTensorB(gmB);
        mm.IterateAll(gmC);
    }
    mm.End();
}
