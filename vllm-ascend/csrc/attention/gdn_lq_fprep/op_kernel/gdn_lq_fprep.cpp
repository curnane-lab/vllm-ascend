#include "kernel_operator.h"

using namespace AscendC;

// M3-3b stage F1 (AIV). Per due (slot, head): assemble the flush staging.
// Due = fill >= L (M3-1 semantics); non-due heads cost one 4 B fill read.
// flags[out] carries the head's fill VALUE (not 0/1): the host wrapper
// predicts next step's due set as fill == L-1 without any host sync.
// ONE cube GEMM (fgemv phase 1) then produces S fp16:
//   S = [p*q^T | ubuf^T | 0 | codes] @ [u ; w*kbuf ; 0 ; diag(sk/127)]
// ab_stage[g] fp16 [192,V]: rows 0-3 p*q (low-rank term is p * q^T u),
//   4-19 ubuf, 20-31 zero, 32-191 = codes rows scaled by (s_v[v]*p)
// ba_stage[g] fp16 [192,K]: rows 0-3 u, 4-19 w_j*kbuf_j, 20-31 zero,
//   32-191 diagonal s_k/127
// All writes contiguous copies / per-row vector ops; no transposes.

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr uint32_t F_R = 4;
constexpr uint32_t F_L = 16;
constexpr uint32_t F_PAD = 192;  // 32 + F_V zero-padded K-dim
constexpr uint32_t F_SCAN = 16;  // batched due-scan width (v1 pattern)
constexpr float QMAX = 127.0f;

constexpr uint32_t HEAD_BYTES = 32768;
constexpr uint32_t OFF_CODES = 0;
constexpr uint32_t OFF_SK = 16384;
constexpr uint32_t OFF_SV = 16896;
constexpr uint32_t OFF_U = 17408;
constexpr uint32_t OFF_Q = 18432;
constexpr uint32_t OFF_KBUF = 19456;
constexpr uint32_t OFF_UBUF = 23552;
constexpr uint32_t OFF_W = 27648;
constexpr uint32_t OFF_P = 27712;
constexpr uint32_t OFF_FILL = 27716;

__aicore__ inline void FprepSyncV2S()
{
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

class GdnLqFprepKernel {
public:
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR idx, GM_ADDR abStage,
                                GM_ADDR flags, GM_ADDR baStage,
                                uint32_t myStart,
                                uint32_t myCount, uint32_t batchNum,
                                uint32_t hNum, int64_t sSlot, TPipe* pipe)
    {
        this->myStart = myStart;
        this->myCount = myCount;
        this->batch = batchNum;
        this->hNum = hNum;
        this->sSlot = sSlot;
        this->pipe = pipe;

        stateRaw = reinterpret_cast<__gm__ uint8_t*>(state);
        gmIdx.SetGlobalBuffer((__gm__ int64_t*)idx);
        abBase = reinterpret_cast<__gm__ half*>(abStage);
        flagsBase = reinterpret_cast<__gm__ int32_t*>(flags);
        baBase = reinterpret_cast<__gm__ half*>(baStage);

        pipe->InitBuffer(codesBuf, F_V * F_K * sizeof(int8_t));
        pipe->InitBuffer(kbufRaw, F_L * F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(ubufRaw, F_L * F_V * sizeof(bfloat16_t));
        pipe->InitBuffer(uqRaw, 2 * F_R * F_V * sizeof(half));
        pipe->InitBuffer(aBuf, F_PAD * F_V * sizeof(half));
        pipe->InitBuffer(bBuf, F_PAD * F_K * sizeof(half));
        pipe->InitBuffer(tmpF32, F_L * F_K * sizeof(float));
        pipe->InitBuffer(halfRow, F_K * sizeof(half));
        pipe->InitBuffer(skF, F_K * sizeof(float));
        pipe->InitBuffer(svF, F_V * sizeof(float));
        pipe->InitBuffer(wF, F_L * sizeof(float));
        pipe->InitBuffer(metaBuf, 3 * 32);
        // scan buffer second half carries the per-instance slot ids
        // for the flags plane 1 write (host mirror keys fills by slot)
        pipe->InitBuffer(scanBuf, 2 * F_SCAN * 8 * sizeof(int32_t));
        pipe->InitBuffer(idxBuf, batchNum * sizeof(int64_t));

        codesUb = codesBuf.Get<int8_t>();
        kbufRawUb = kbufRaw.Get<bfloat16_t>();
        ubufRawUb = ubufRaw.Get<bfloat16_t>();
        uqRawUb = uqRaw.Get<half>();
        aUb = aBuf.Get<half>();
        bUb = bBuf.Get<half>();
        tmpF32Ub = tmpF32.Get<float>();
        halfUb = halfRow.Get<half>();
        skUb = skF.Get<float>();
        svUb = svF.Get<float>();
        wUb = wF.Get<float>();
        pUb = metaBuf.Get<float>()[16];
        fillScanUb = scanBuf.Get<int32_t>();
        slotScanUb = scanBuf.Get<int32_t>()[F_SCAN * 8];
        idxUb = idxBuf.Get<int64_t>();

        DataCopyPadParams pad{false, 0, 0, 0};
        this->padParams = pad;
    }

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
        // Due scan with batched fill loads (gdn_lq_flush v1 pattern):
        // issue a whole batch of 4 B GM reads back-to-back, then ONE
        // PipeBarrier, then consume. flags plane 0 [inst] are the raw
        // fill values copied straight out of the scan buffer; plane 1
        // [numInstances + inst] carries the slot id, so the host mirror
        // keys fills BY SLOT (a batch position swap must not inherit a
        // stale fill - E2). slotScanUb is filled by an S-pipe SetValue
        // in the load loop, ordered before the MTE3 stores by the
        // PipeBarriers.
        uint32_t numInst = batch * hNum;
        for (uint32_t s0 = myStart; s0 < myStart + myCount; s0 += F_SCAN) {
            uint32_t n = (myStart + myCount - s0) < F_SCAN
                             ? (myStart + myCount - s0)
                             : F_SCAN;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t inst = s0 + i;
                uint32_t b = inst / hNum;
                uint32_t h = inst % hNum;
                int32_t slot =
                    idxUb.ReinterpretCast<int32_t>().GetValue(b * 2);
                slotScanUb.SetValue(i * 8, slot);
                __gm__ uint8_t* base = stateRaw
                    + static_cast<uint64_t>(slot) * static_cast<uint64_t>(sSlot)
                    + static_cast<uint64_t>(h) * HEAD_BYTES;
                GlobalTensor<int32_t> gmFill;
                gmFill.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int32_t*>(base + OFF_FILL));
                DataCopyPad(fillScanUb[i * 8], gmFill[0],
                            DataCopyParams{1, 4, 0, 0}, padParams);
            }
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t inst = s0 + i;
                GlobalTensor<int32_t> gmFlag;
                gmFlag.SetGlobalBuffer(flagsBase + inst);
                DataCopyPad(gmFlag[0], fillScanUb[i * 8],
                            DataCopyParams{1, 4, 0, 0});
                GlobalTensor<int32_t> gmSlot;
                gmSlot.SetGlobalBuffer(flagsBase + numInst + inst);
                DataCopyPad(gmSlot[0], slotScanUb[i * 8],
                            DataCopyParams{1, 4, 0, 0});
                if (fillScanUb.GetValue(i * 8)
                    >= static_cast<int32_t>(F_L)) {
                    StageOne(inst);
                }
            }
            // scan buffer slots are reused by the next batch's loads
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void StageOne(uint32_t inst)
    {
        uint32_t b = inst / hNum;
        uint32_t h = inst % hNum;
        int32_t slot = idxUb.ReinterpretCast<int32_t>().GetValue(b * 2);
        __gm__ uint8_t* base = stateRaw
            + static_cast<uint64_t>(slot) * static_cast<uint64_t>(sSlot)
            + static_cast<uint64_t>(h) * HEAD_BYTES;
        // due check + flags write happen in the batched scan (Process);
        // here: stage the due head's A/B rows for the cube GEMM.
        GlobalTensor<float> gmP;
        gmP.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + OFF_P));
        DataCopyPad(pUb, gmP[0], DataCopyParams{1, 4, 0, 0}, padParams);

        GlobalTensor<int8_t> gmCodes;
        gmCodes.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base + OFF_CODES));
        GlobalTensor<bfloat16_t> gmKbuf;
        gmKbuf.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(base + OFF_KBUF));
        GlobalTensor<bfloat16_t> gmUbuf;
        gmUbuf.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(base + OFF_UBUF));
        GlobalTensor<half> gmU;
        gmU.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(base + OFF_U));
        GlobalTensor<half> gmQ;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(base + OFF_Q));
        GlobalTensor<float> gmSk;
        gmSk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + OFF_SK));
        GlobalTensor<float> gmSv;
        gmSv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + OFF_SV));
        GlobalTensor<float> gmW;
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + OFF_W));

        DataCopy(codesUb, gmCodes[0], F_V * F_K);
        DataCopy(kbufRawUb, gmKbuf[0], F_L * F_K);
        DataCopy(ubufRawUb, gmUbuf[0], F_L * F_K);
        DataCopy(uqRawUb, gmU[0], F_R * F_V);
        DataCopy(uqRawUb[F_R * F_V], gmQ[0], F_R * F_V);
        DataCopy(skUb, gmSk[0], F_K);
        DataCopy(svUb, gmSv[0], F_V);
        DataCopy(wUb, gmW[0], F_L);
        PipeBarrier<PIPE_ALL>();  // MTE2 loads -> V/S consumers
        float pval = pUb.GetValue(0);

        // ---- A staging [192,V] fp16: rows 0-3 = p*q (contract: the
        // low-rank term is p * q^T u; q is scaled in fp32 then rounded to
        // fp16, matching the torch reference's fp32 p * (fp16 q) path as
        // closely as an fp16 staging allows), 4-19 ubuf, 20-31 zero,
        // rows 32-159 = diag(s_v*p) (one scalar write per row), rest zero
        Duplicate(aUb, static_cast<half>(0.0f), F_PAD * F_V);
        Cast(tmpF32Ub, uqRawUb[F_R * F_V], RoundMode::CAST_NONE, F_R * F_V);
        Muls(tmpF32Ub, tmpF32Ub, pval, F_R * F_V);
        Cast(aUb, tmpF32Ub, RoundMode::CAST_NONE, F_R * F_V);
        Cast(tmpF32Ub, ubufRawUb, RoundMode::CAST_NONE, F_L * F_V);
        Cast(aUb[4 * F_V], tmpF32Ub, RoundMode::CAST_NONE, F_L * F_V);
        FprepSyncV2S();
        for (uint32_t v = 0; v < F_V; ++v) {
            float av = svUb.GetValue(v) * pval;
            aUb.SetValue((32 + v) * F_V + v, static_cast<half>(av));
        }
        // ---- B staging [192,K] fp16: rows 0-3 u, 4-19 w_j*kbuf_j,
        // 20-31 zero, 32-159 = codes rows scaled by s_k/127, rest zero
        Duplicate(bUb, static_cast<half>(0.0f), F_PAD * F_K);
        DataCopy(bUb, uqRawUb, F_R * F_V);
        for (uint32_t j = 0; j < F_L; ++j) {
            float wj = wUb.GetValue(j);
            Cast(tmpF32Ub, kbufRawUb[j * F_K], RoundMode::CAST_NONE, F_K);
            Muls(tmpF32Ub, tmpF32Ub, wj, F_K);
            Cast(bUb[(4 + j) * F_K], tmpF32Ub, RoundMode::CAST_NONE, F_K);
        }
        for (uint32_t v = 0; v < F_V; ++v) {
            // codes row v (int8 -> fp16 exact) scaled column-wise by sk/127
            Cast(halfUb, codesUb[v * F_K], RoundMode::CAST_NONE, F_K);
            Cast(tmpF32Ub, halfUb, RoundMode::CAST_NONE, F_K);
            Mul(tmpF32Ub, tmpF32Ub, skUb, F_K);
            Muls(tmpF32Ub, tmpF32Ub, 1.0f / QMAX, F_K);
            Cast(bUb[(32 + v) * F_K], tmpF32Ub, RoundMode::CAST_NONE, F_K);
        }

        uint64_t g = inst;
        GlobalTensor<half> gmAB;
        gmAB.SetGlobalBuffer(abBase + g * F_PAD * F_V);
        GlobalTensor<half> gmBA;
        gmBA.SetGlobalBuffer(baBase + g * F_PAD * F_K);
        DataCopy(gmAB, aUb, F_PAD * F_V);
        DataCopy(gmBA, bUb, F_PAD * F_K);
        // drain the staging reads before the next instance reuses the UB
        AscendC::TEventID e2 = pipe->FetchEventID(HardEvent::MTE3_V);
        SetFlag<HardEvent::MTE3_V>(e2);
        WaitFlag<HardEvent::MTE3_V>(e2);
    }

    __gm__ uint8_t* stateRaw = nullptr;
    GlobalTensor<int64_t> gmIdx;
    __gm__ half* abBase = nullptr;
    __gm__ int32_t* flagsBase = nullptr;
    __gm__ half* baBase = nullptr;
    DataCopyPadParams padParams{false, 0, 0, 0};
    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> codesBuf, kbufRaw, ubufRaw, uqRaw;
    TBuf<QuePosition::VECCALC> aBuf, bBuf, tmpF32, halfRow, skF, svF, wF;
    TBuf<QuePosition::VECCALC> metaBuf, scanBuf, idxBuf;
    LocalTensor<int8_t> codesUb;
    LocalTensor<bfloat16_t> kbufRawUb, ubufRawUb;
    LocalTensor<half> uqRawUb;
    LocalTensor<half> aUb, bUb;
    LocalTensor<float> tmpF32Ub;
    LocalTensor<half> halfUb;
    LocalTensor<float> skUb, svUb, wUb;
    LocalTensor<int32_t> fillScanUb, slotScanUb;
    LocalTensor<float> pUb;
    LocalTensor<int64_t> idxUb;
    uint32_t myStart = 0, myCount = 0, batch = 1, hNum = 32;
    int64_t sSlot = 0;
};

extern "C" __global__ __aicore__ void gdn_lq_fprep(
    GM_ADDR state, GM_ADDR idx, GM_ADDR ab_stage, GM_ADDR flags,
    GM_ADDR ba_stage, GM_ADDR workspace, GM_ADDR tiling)
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
    GdnLqFprepKernel op;
    op.Init(state, idx, ab_stage, flags, ba_stage, myStart, myCount,
            tilingData.numInstances / tilingData.h, tilingData.h,
            tilingData.sSlot, &pipe);
    op.LoadIdx();
    op.Process();
}
