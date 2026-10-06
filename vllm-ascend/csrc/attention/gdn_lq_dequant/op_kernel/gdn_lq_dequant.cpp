#include "kernel_operator.h"

using namespace AscendC;

// M2-D2 step v2, stage 1 (AIV-only). Per (slot, value-head) instance:
//   staging[inst] = fp16(codes[slot, head])                  [V,K] exact
//   aq[inst,0,:]  = fp16(kn[tok, head//rep] * s_k / 127)
//   aq[inst,1,:]  = fp16(qn[tok, head//rep] * scale * s_k / 127)
//   aq[inst,2:,:] = 0
// The /127 and the K**-0.5 (scale) fold into the fp32 premultiplies so the
// cube (stage 2) sees pure fp16 operands and the epilogue (stage 3) consumes
// C[.,0]/C[.,1] directly as the v1 codesKn / codesQn row dots.
// Fixed geometry: V=K=128, composite per-head block layout (GdnLqStep).

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr float QMAX = 127.0f;

// per-(slot, head) contiguous 32 KB block layout (gdn_lq_step / M1)
constexpr uint32_t HEAD_BYTES = 32768;
constexpr uint32_t OFF_CODES = 0;
constexpr uint32_t OFF_SK = 16384;

class GdnLqDequantKernel {
public:
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR idx, GM_ADDR kn,
                                GM_ADDR qn, GM_ADDR staging, GM_ADDR aq,
                                uint32_t myStart, uint32_t myCount,
                                uint32_t batchNum, uint32_t hNum,
                                uint32_t nkNum, int64_t sSlot,
                                float scale, TPipe* pipe)
    {
        this->myStart = myStart;
        this->myCount = myCount;
        this->batch = batchNum;
        this->hNum = hNum;
        this->rep = hNum / nkNum;
        this->sSlot = sSlot;
        this->scale = scale;
        this->pipe = pipe;

        stateRaw = reinterpret_cast<__gm__ uint8_t*>(state);
        gmIdx.SetGlobalBuffer((__gm__ int64_t*)idx);
        knBase = reinterpret_cast<__gm__ bfloat16_t*>(kn);
        qnBase = reinterpret_cast<__gm__ bfloat16_t*>(qn);
        stagingBase = reinterpret_cast<__gm__ half*>(staging);
        aqBase = reinterpret_cast<__gm__ half*>(aq);

        pipe->InitBuffer(codesIBuf, F_V * F_K * sizeof(int8_t));
        pipe->InitBuffer(codesHBuf, F_V * F_K * sizeof(half));
        pipe->InitBuffer(kqvBuf, 2 * F_K * sizeof(bfloat16_t));  // kn, qn rows
        pipe->InitBuffer(skBuf, F_K * sizeof(float));
        pipe->InitBuffer(aqFBuf, 16 * F_K * sizeof(float));
        pipe->InitBuffer(aqHBuf, 16 * F_K * sizeof(half));
        pipe->InitBuffer(idxBuf, batchNum * sizeof(int64_t));

        codesIUb = codesIBuf.Get<int8_t>();
        codesHUb = codesHBuf.Get<half>();
        knRawUb = kqvBuf.Get<bfloat16_t>();
        qnRawUb = kqvBuf.Get<bfloat16_t>()[F_K];
        skUb = skBuf.Get<float>();
        aqFUb = aqFBuf.Get<float>();
        aqHUb = aqHBuf.Get<half>();
        idxUb = idxBuf.Get<int64_t>();

        DataCopyPadParams pad{false, 0, 0, 0};
        this->padParams = pad;
    }

    // slot ids for the whole batch, loaded once (batch <= 64)
    __aicore__ inline void LoadIdx()
    {
        DataCopyPad(idxUb, gmIdx[0],
                    DataCopyParams{1, static_cast<uint16_t>(batch * 8), 0, 0},
                    padParams);
        AscendC::TEventID evt = pipe->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(evt);
        WaitFlag<HardEvent::MTE2_V>(evt);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }

private:
    __aicore__ inline void ProcessOne(uint32_t inst)
    {
        uint32_t b = inst / hNum;
        uint32_t h = inst % hNum;
        uint32_t kh = h / rep;
        int32_t slot = idxUb.ReinterpretCast<int32_t>().GetValue(b * 2);
        __gm__ uint8_t* headBase = stateRaw
            + static_cast<uint64_t>(slot) * static_cast<uint64_t>(sSlot)
            + static_cast<uint64_t>(h) * HEAD_BYTES;

        uint64_t instOff = static_cast<uint64_t>(inst);
        GlobalTensor<int8_t> gmCodes;
        gmCodes.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(headBase
                                                                  + OFF_CODES));
        GlobalTensor<float> gmSk;
        gmSk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(headBase
                                                             + OFF_SK));
        GlobalTensor<bfloat16_t> gmKnRow;
        gmKnRow.SetGlobalBuffer(knBase
            + (static_cast<uint64_t>(b) * (hNum / rep) + kh) * F_K);
        GlobalTensor<bfloat16_t> gmQnRow;
        gmQnRow.SetGlobalBuffer(qnBase
            + (static_cast<uint64_t>(b) * (hNum / rep) + kh) * F_K);

        DataCopy(codesIUb, gmCodes[0], F_V * F_K);
        DataCopy(skUb, gmSk[0], F_K);
        DataCopy(knRawUb, gmKnRow[0], F_K);
        DataCopy(qnRawUb, gmQnRow[0], F_K);
        AscendC::TEventID evtMte2V = pipe->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(evtMte2V);
        WaitFlag<HardEvent::MTE2_V>(evtMte2V);

        // staging: row-wise 128-element int8->fp16 casts (proven repeat
        // budget), then one 32 KB store per instance
        for (uint32_t v = 0; v < F_V; ++v) {
            Cast(codesHUb[v * F_K], codesIUb[v * F_K], RoundMode::CAST_NONE,
                 F_K);
        }
        // aq rows 0/1: fp32 premultiply then cast; pad rows stay zero
        Cast(aqFUb, knRawUb, RoundMode::CAST_NONE, F_K);
        Mul(aqFUb, aqFUb, skUb, F_K);
        Muls(aqFUb, aqFUb, 1.0f / QMAX, F_K);
        Cast(aqFUb[F_K], qnRawUb, RoundMode::CAST_NONE, F_K);
        Mul(aqFUb[F_K], aqFUb[F_K], skUb, F_K);
        Muls(aqFUb[F_K], aqFUb[F_K], scale / QMAX, F_K);
        Duplicate(aqFUb[2 * F_K], 0.0f, 14 * F_K);
        for (uint32_t m = 0; m < 16; ++m) {
            Cast(aqHUb[m * F_K], aqFUb[m * F_K], RoundMode::CAST_NONE, F_K);
        }

        GlobalTensor<half> gmStageOut;
        gmStageOut.SetGlobalBuffer(stagingBase + instOff * F_V * F_K);
        GlobalTensor<half> gmAqOut;
        gmAqOut.SetGlobalBuffer(aqBase + instOff * 16 * F_K);
        DataCopy(gmStageOut, codesHUb, F_V * F_K);
        DataCopy(gmAqOut, aqHUb, 16 * F_K);
        // drain the tile reads before the next instance reuses the UB
        AscendC::TEventID evtMte3V = pipe->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(evtMte3V);
        WaitFlag<HardEvent::MTE3_V>(evtMte3V);
    }
    __gm__ uint8_t* stateRaw = nullptr;
    GlobalTensor<int64_t> gmIdx;
    __gm__ bfloat16_t* knBase = nullptr;
    __gm__ bfloat16_t* qnBase = nullptr;
    __gm__ half* stagingBase = nullptr;
    __gm__ half* aqBase = nullptr;
    TBuf<QuePosition::VECCALC> codesIBuf, codesHBuf, kqvBuf, skBuf, aqFBuf;
    TBuf<QuePosition::VECCALC> aqHBuf, idxBuf;
    LocalTensor<int8_t> codesIUb;
    LocalTensor<half> codesHUb;
    LocalTensor<bfloat16_t> knRawUb, qnRawUb;
    LocalTensor<float> skUb;
    LocalTensor<float> aqFUb;
    LocalTensor<half> aqHUb;
    LocalTensor<int64_t> idxUb;
    DataCopyPadParams padParams{false, 0, 0, 0};
    TPipe* pipe = nullptr;
    uint32_t myStart = 0, myCount = 0, batch = 1, hNum = 32, rep = 2;
    int64_t sSlot = 0;
    float scale = 1.0f;
};

extern "C" __global__ __aicore__ void gdn_lq_dequant(
    GM_ADDR state, GM_ADDR idx, GM_ADDR kn, GM_ADDR qn, GM_ADDR staging,
    GM_ADDR aq, GM_ADDR workspace, GM_ADDR tiling)
{
    ENABLE_FEATURE_FOR_COMPILE(default, KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() >= tilingData.usedCoreNum) {
        return;
    }
    uint32_t myStart;
    uint32_t myCount;
    if (GetBlockIdx() < tilingData.remainderCores) {
        myStart = GetBlockIdx() * (tilingData.instancesPerCore + 1);
        myCount = tilingData.instancesPerCore + 1;
    } else {
        myStart = tilingData.remainderCores
            * (tilingData.instancesPerCore + 1)
            + (GetBlockIdx() - tilingData.remainderCores)
            * tilingData.instancesPerCore;
        myCount = tilingData.instancesPerCore;
    }
    TPipe pipe;
    GdnLqDequantKernel op;
    op.Init(state, idx, kn, qn, staging, aq, myStart, myCount,
            tilingData.numInstances / tilingData.h, tilingData.h,
            tilingData.nk, tilingData.sSlot, tilingData.scale, &pipe);
    op.LoadIdx();
    op.Process();
}
