#include "kernel_operator.h"

using namespace AscendC;

// M3-3b AIV tail, attr phase:
//   1: qk  = mgs(c_stage cols 0..3 [K,4]) -> kq_stage [16,K] fp16
//   2: qv' = mgs(c_stage cols 0..3 [V,4]) -> vq_stage [16,V] fp16
//      (vq_stage is an INPUT here so one output tensor serves both phases:
//       phase 2 writes it in place via the aliased input pointer)
//   3: E = S - qv' uk^T; sk/sv/codes; checkpoint writeback + w=0, p=1,
//      fill=L (M3-1 semantics). Due heads only.
// Mgs4 / reduce helpers replicate gdn_lq_flush exactly (same torch
// reference); scalar-heavy but tiny compared to the retired vector rebuild.

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr uint32_t F_R = 4;
constexpr uint32_t F_L = 16;
constexpr float QMAX = 127.0f;
constexpr float QEPS = 1e-12f;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;

constexpr uint32_t HEAD_BYTES = 32768;
constexpr uint32_t OFF_CODES = 0;
constexpr uint32_t OFF_SK = 16384;
constexpr uint32_t OFF_SV = 16896;
constexpr uint32_t OFF_U = 17408;
constexpr uint32_t OFF_Q = 18432;
constexpr uint32_t OFF_W = 27648;
constexpr uint32_t OFF_P = 27712;
constexpr uint32_t OFF_FILL = 27716;

__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

// slot[0] = sum of src[0..count); src clobbered; repeat=1 single-shot form
// with the mask reset immediately before (gdn_lq_step finding).
__aicore__ inline void ReduceSumHalfInterval(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
    int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / 2;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
        }
    }
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
    WholeReduceSum(dst_local, src_local, ELEM_PER_REP_FP32, 1, 1, 1, 8);
#else
    WholeReduceSum<float, false>(dst_local, src_local, ELEM_PER_REP_FP32, 1,
                                 1, 1, 8);
#endif
}

// |src[0..8]| max, in place; src clobbered (same helper as gdn_lq_flush)
__aicore__ inline void ReduceMaxToEight(const LocalTensor<float>& src_local,
                                        int32_t count)
{
    int32_t bodyCount = count;
    while (bodyCount > 8) {
        bodyCount = bodyCount / 2;
        Max(src_local, src_local, src_local[bodyCount], bodyCount);
    }
}

class GdnLqFtailKernel {
public:
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR idx, GM_ADDR sStage,
                                GM_ADDR cStage, GM_ADDR vqStage,
                                GM_ADDR kqStage, uint32_t myStart,
                                uint32_t myCount, uint32_t batchNum,
                                uint32_t hNum, int64_t sSlot, uint32_t phase,
                                TPipe* pipe)
    {
        this->myStart = myStart;
        this->myCount = myCount;
        this->batch = batchNum;
        this->hNum = hNum;
        this->sSlot = sSlot;
        this->phase = phase;
        this->pipe = pipe;

        stateRaw = reinterpret_cast<__gm__ uint8_t*>(state);
        gmIdx.SetGlobalBuffer((__gm__ int64_t*)idx);
        sBase = reinterpret_cast<__gm__ uint8_t*>(sStage);
        cBase = reinterpret_cast<__gm__ float*>(cStage);
        vqBase = reinterpret_cast<__gm__ half*>(vqStage);
        kqBase = reinterpret_cast<__gm__ half*>(kqStage);

        pipe->InitBuffer(sBuf, F_V * F_K * sizeof(float));
        pipe->InitBuffer(sHBuf, F_V * F_K * sizeof(half));
        pipe->InitBuffer(cBuf, F_V * 32 * sizeof(float));
        pipe->InitBuffer(vecBuf, F_R * F_K * sizeof(float));
        pipe->InitBuffer(outQ, 1, F_V * F_K + 3584);
        pipe->InitBuffer(scrBuf, 4 * F_K * sizeof(float));
        pipe->InitBuffer(redBuf, 16 * 8 * sizeof(float));
        pipe->InitBuffer(halfBuf, F_K * sizeof(half));
        pipe->InitBuffer(uhBuf, F_V * 32 * sizeof(half));
        pipe->InitBuffer(skF, F_K * sizeof(float));
        pipe->InitBuffer(svF, F_V * sizeof(float));
        pipe->InitBuffer(tmpBuf, 2 * F_K * sizeof(float));
        pipe->InitBuffer(idxBuf, batchNum * sizeof(int64_t));
        pipe->InitBuffer(metaBuf, 64);
        pipe->InitBuffer(codesBuf, F_V * F_K * sizeof(int8_t));

        sUb = sBuf.Get<float>();
        sHUb = sHBuf.Get<half>();
        cUb = cBuf.Get<float>();
        vecUb = vecBuf.Get<float>();
        scrUb = scrBuf.Get<float>();
        redUb = redBuf.Get<float>();
        halfUb = halfBuf.Get<half>();
        uhUb = uhBuf.Get<half>();
        skUb = skF.Get<float>();
        svUb = svF.Get<float>();
        tmpUb = tmpBuf.Get<float>();
        tmp2Ub = tmpBuf.Get<float>()[F_K];
        idxUb = idxBuf.Get<int64_t>();
        fillUb = metaBuf.Get<int32_t>();
        codesUb = codesBuf.Get<int8_t>();

        DataCopyPadParams pad{false, 0, 0, 0};
        this->padParams = pad;
    }

    __aicore__ inline void LoadIdx()
    {
        DataCopyPad(idxUb, gmIdx[0],
                    DataCopyParams{1, static_cast<uint16_t>(batch * 8), 0, 0},
                    padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }

private:
    __aicore__ inline void SyncV2S()
    {
        SetFlag<HardEvent::V_S>(FTAIL_V_S_EVT);
        WaitFlag<HardEvent::V_S>(FTAIL_V_S_EVT);
    }

    __aicore__ inline void SyncS2V()
    {
        SetFlag<HardEvent::S_V>(FTAIL_S_V_EVT);
        WaitFlag<HardEvent::S_V>(FTAIL_S_V_EVT);
    }

    __aicore__ inline float DotV(const LocalTensor<float>& a,
                                 const LocalTensor<float>& b)
    {
        Mul(tmpUb, a, b, F_K);
        ReduceSumHalfInterval(redUb, tmpUb, F_K);
        SyncV2S();
        return redUb.GetValue(0);
    }

    // mgs on 4 [F_K] vectors in vec (replicates gdn_lq_flush::Mgs4)
    __aicore__ inline void Mgs4(const LocalTensor<float>& vec)
    {
        float d = DotV(vec, vec);
        float inv = 1.0f / sqrt(d >= QEPS ? d : QEPS);
        Muls(vec, vec, inv, F_K);
        for (uint32_t r = 1; r < F_R; ++r) {
            LocalTensor<float> vr = vec[r * F_K];
            for (uint32_t pass = 0; pass < 2; ++pass) {
                for (uint32_t c = 0; c < r; ++c) {
                    float coef = DotV(vec[c * F_K], vr);
                    Muls(tmpUb, vec[c * F_K], coef, F_K);
                    Sub(vr, vr, tmpUb, F_K);
                }
            }
            float dr = DotV(vr, vr);
            float invr = 1.0f / sqrt(dr >= QEPS ? dr : QEPS);
            Muls(vr, vr, invr, F_K);
        }
    }

    __aicore__ inline bool CheckDue(uint32_t inst, __gm__ uint8_t** headOut)
    {
        uint32_t b = inst / hNum;
        uint32_t h = inst % hNum;
        int32_t slot = idxUb.ReinterpretCast<int32_t>().GetValue(b * 2);
        __gm__ uint8_t* head = stateRaw
            + static_cast<uint64_t>(slot) * static_cast<uint64_t>(sSlot)
            + static_cast<uint64_t>(h) * HEAD_BYTES;
        *headOut = head;
        GlobalTensor<int32_t> gmFill;
        gmFill.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(head + OFF_FILL));
        DataCopyPad(fillUb, gmFill[0], DataCopyParams{1, 4, 0, 0}, padParams);
        PipeBarrier<PIPE_ALL>();
        return fillUb.GetValue(0) >= static_cast<int32_t>(F_L);
    }

    __aicore__ inline void ProcessOne(uint32_t inst)
    {
        __gm__ uint8_t* head = nullptr;
        if (!CheckDue(inst, &head)) {
            return;
        }
        uint64_t g = inst;
        GlobalTensor<float> gmC;
        gmC.SetGlobalBuffer(cBase + g * F_V * 32);

        if (phase == 1 || phase == 2) {
            // load c_stage [128,16] and extract cols 0..3 into 4 vectors
            DataCopy(cUb, gmC[0], F_V * 32);
            PipeBarrier<PIPE_ALL>();
            Duplicate(vecUb, 0.0f, F_R * F_K);
            for (uint32_t k = 0; k < F_K; ++k) {
                for (uint32_t r = 0; r < F_R; ++r) {
                    vecUb.SetValue(r * F_K + k, cUb.GetValue(k * 32 + r));
                }
            }
            SyncS2V();  // S-pipe SetValue writes -> V reads
            Mgs4(vecUb);
            // write fp16 [16,128]: rows 0..3 result, 4..15 zero
            // transposed layout [K,32]: element [k,r] = result[r,k]; the
            // cube then reads B directly (transposed-B ND reads fault at
            // G>64 on this stack). Build in fp32 first (cUb is free after
            // the extraction), then one cast - runtime-indexed half
            // SetValue miscompiles on this toolchain.
            Duplicate(cUb, 0.0f, F_K * 32);
            for (uint32_t k = 0; k < F_K; ++k) {
                for (uint32_t r = 0; r < F_R; ++r) {
                    cUb.SetValue(k * 32 + r, vecUb.GetValue(r * F_K + k));
                }
            }
            SyncS2V();
            // store through the TQue payload path (same proven pattern as
            // gdn_lq_flush / gdn_lq_step; raw TBuf -> GM DataCopy of this
            // tensor faults on this stack)
            LocalTensor<half> oqh = outQ.AllocTensor<half>();
            Cast(oqh, cUb, RoundMode::CAST_NONE, F_K * 32);
            outQ.EnQue(oqh);
            LocalTensor<half> oqd = outQ.DeQue<half>();
            __gm__ half* base = (phase == 1) ? kqBase : vqBase;
            GlobalTensor<half> gmOut;
            gmOut.SetGlobalBuffer(base + g * F_K * 32);
            DataCopy(gmOut, oqd, F_K * 32);
            outQ.FreeTensor(oqd);
            return;
        }

        // ---- phase 3: quantize ----
        GlobalTensor<half> gmS;
        gmS.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(sBase) + g * F_V * F_K);
        GlobalTensor<half> gmVQ;
        gmVQ.SetGlobalBuffer(vqBase + g * 32 * F_V);
        DataCopy(sHUb, gmS[0], F_V * F_K);
        DataCopy(cUb, gmC[0], F_V * 32);
        DataCopy(uhUb, gmVQ[0], F_V * 32);  // qv' transposed [V,32]
        PipeBarrier<PIPE_ALL>();
        for (uint32_t v = 0; v < F_V; ++v) {
            Cast(sUb[v * F_K], sHUb[v * F_K], RoundMode::CAST_NONE, F_K);
        }

        // E = S - sum_r qv[r,v] * uk[r,:]; uk[k,r] = cUb[k*16+r].
        // stage uk rows contiguously first (4 rows x 128 scalars).
        // sumAbs MUST be zero-initialized before the accumulation (v1
        // gdn_lq_flush line ~400 does); without it sk/sv inherited
        // uninitialized UB bytes - intermittently NaN - while codes
        // survived only because Maxs/Mins clamp NaN to +-QMAX.
        LocalTensor<float> sumAbs = scrUb;  // F_K floats, dead after sk
        Duplicate(sumAbs, 0.0f, F_K);
        for (uint32_t r = 0; r < F_R; ++r) {
            for (uint32_t k = 0; k < F_K; ++k) {
                vecUb.SetValue(r * F_K + k, cUb.GetValue(k * 32 + r));
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t v = 0; v < F_V; ++v) {
            LocalTensor<float> srow = sUb[v * F_K];
            float c0 = static_cast<float>(uhUb.GetValue(v * 32 + 0));
            float c1 = static_cast<float>(uhUb.GetValue(v * 32 + 1));
            float c2 = static_cast<float>(uhUb.GetValue(v * 32 + 2));
            float c3 = static_cast<float>(uhUb.GetValue(v * 32 + 3));
            Muls(tmpUb, vecUb[0 * F_K], c0, F_K);
            Muls(tmp2Ub, vecUb[1 * F_K], c1, F_K);
            Add(tmpUb, tmpUb, tmp2Ub, F_K);
            Muls(tmp2Ub, vecUb[2 * F_K], c2, F_K);
            Add(tmpUb, tmpUb, tmp2Ub, F_K);
            Muls(tmp2Ub, vecUb[3 * F_K], c3, F_K);
            Add(tmpUb, tmpUb, tmp2Ub, F_K);
            Sub(srow, srow, tmpUb, F_K);
            Abs(tmpUb, srow, F_K);
            Add(sumAbs, sumAbs, tmpUb, F_K);
        }
        Muls(skUb, sumAbs, 0.0078125f, F_K);  // 1/V
        Sqrt(skUb, skUb, F_K);
        Maxs(skUb, skUb, QEPS, F_K);
        SyncV2S();

        // sv[v] = max_k |E|/sk; codes = rint(127 E / (sk sv))
        for (uint32_t v0 = 0; v0 < F_V; v0 += 4) {
            for (uint32_t i = 0; i < 4; ++i) {
                LocalTensor<float> acc = scrUb[i * F_K];
                Div(acc, sUb[(v0 + i) * F_K], skUb, F_K);
                Abs(acc, acc, F_K);
                ReduceMaxToEight(acc, F_K);
            }
            SyncV2S();
            for (uint32_t i = 0; i < 4; ++i) {
                uint32_t v = v0 + i;
                float svv = scrUb[i * F_K].GetValue(0);
                for (uint32_t j = 1; j < 8; ++j) {
                    float c = scrUb[i * F_K].GetValue(j);
                    svv = c > svv ? c : svv;
                }
                if (svv < QEPS) {
                    svv = QEPS;
                }
                svUb.SetValue(v, svv);
            }
        }
        PipeBarrier<PIPE_V>();
        SyncV2S();
        for (uint32_t v = 0; v < F_V; ++v) {
            float svv = svUb.GetValue(v);
            float inv = QMAX / svv;
            Div(tmpUb, sUb[v * F_K], skUb, F_K);
            Muls(tmpUb, tmpUb, inv, F_K);
            Maxs(tmpUb, tmpUb, -QMAX, F_K);
            Mins(tmpUb, tmpUb, QMAX, F_K);
            Cast(halfUb, tmpUb, RoundMode::CAST_RINT, F_K);
            Cast(codesUb[v * F_K], halfUb, RoundMode::CAST_RINT, F_K);
        }

        // ---- payload writeback (same layout as gdn_lq_flush) ----
        // V/S-computed skUb/svUb/codesUb/vecUb are consumed by the UB->UB
        // payload copies below; without a full barrier the copies
        // intermittently read un-settled data (s_k/s_v came back NaN on
        // back-to-back cached launches while codes survived only because
        // the scalar q-gather loop delayed its copy).
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int8_t> oq = outQ.AllocTensor<int8_t>();
        DataCopy(oq, codesUb, F_V * F_K);
        LocalTensor<float> sf = oq.ReinterpretCast<float>();
        DataCopy(sf[F_V * F_K / 4], skUb, F_K);
        DataCopy(sf[F_V * F_K / 4 + F_K], svUb, F_V);
        // u_new[r,k] = uk[r,k] as fp16 (vecUb holds the uk rows)
        Cast(oq.ReinterpretCast<half>()[F_R * F_V * 17], vecUb,
             RoundMode::CAST_RINT, F_R * F_K);
        // q_new[r,v] = qv'[r,v]: gather from the transposed [V,32] staging
        for (uint32_t r = 0; r < F_R; ++r) {
            for (uint32_t v = 0; v < F_V; ++v) {
                half qv = uhUb.GetValue(v * 32 + r);
                oq.ReinterpretCast<half>().SetValue(F_R * F_V * 18 + r * F_V + v,
                                                    qv);
            }
        }
        Duplicate(sf[F_V * F_K / 4 + 6 * F_K], 0.0f, F_L);
        Duplicate(sf[F_V * F_K / 4 + 6 * F_K + 16], 1.0f, 8);
        Duplicate(oq.ReinterpretCast<int32_t>()[F_V * F_K / 4 + 6 * F_K + 24],
                  static_cast<int32_t>(F_L), 1);
        outQ.EnQue(oq);
        LocalTensor<int8_t> dq = outQ.DeQue<int8_t>();
        GlobalTensor<int8_t> gmCodes;
        gmCodes.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(head + OFF_CODES));
        DataCopy(gmCodes[0], dq, F_V * F_K);
        LocalTensor<float> df = dq.ReinterpretCast<float>();
        GlobalTensor<float> gmSk;
        gmSk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(head + OFF_SK));
        GlobalTensor<float> gmSv;
        gmSv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(head + OFF_SV));
        DataCopy(gmSk[0], df[F_V * F_K / 4], F_K);
        DataCopy(gmSv[0], df[F_V * F_K / 4 + F_K], F_V);
        LocalTensor<half> dh = dq.ReinterpretCast<half>();
        GlobalTensor<half> gmU;
        gmU.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(head + OFF_U));
        GlobalTensor<half> gmQ;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(head + OFF_Q));
        // u from the payload staging, q gathered into the payload above
        DataCopy(gmU[0], dh[F_R * F_V * 17], F_R * F_V);
        DataCopy(gmQ[0], dh[F_R * F_V * 18], F_R * F_V);
        GlobalTensor<float> gmW;
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(head + OFF_W));
        DataCopy(gmW[0], df[F_V * F_K / 4 + 6 * F_K], F_L);
        GlobalTensor<float> gmP;
        gmP.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(head + OFF_P));
        DataCopyPad(gmP[0], df[F_V * F_K / 4 + 6 * F_K + 16],
                    DataCopyParams{1, 4, 0, 0});
        GlobalTensor<int32_t> gmFill;
        gmFill.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(head + OFF_FILL));
        DataCopyPad(gmFill[0],
                    dq.ReinterpretCast<int32_t>()[F_V * F_K / 4 + 6 * F_K + 24],
                    DataCopyParams{1, 4, 0, 0});
        outQ.FreeTensor(dq);
    }

    static constexpr uint32_t FTAIL_V_S_EVT = 2;
    static constexpr uint32_t FTAIL_S_V_EVT = 3;
    static constexpr uint32_t FTAIL_V_MTE3_EVT = 4;
    __gm__ uint8_t* stateRaw = nullptr;
    GlobalTensor<int64_t> gmIdx;
    __gm__ uint8_t* sBase = nullptr;
    __gm__ float* cBase = nullptr;
    __gm__ half* vqBase = nullptr;
    __gm__ half* kqBase = nullptr;
    DataCopyPadParams padParams{false, 0, 0, 0};
    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> sBuf, sHBuf, cBuf, vecBuf, scrBuf, redBuf;
    TBuf<QuePosition::VECCALC> halfBuf, uhBuf, skF, svF, tmpBuf;
    TBuf<QuePosition::VECCALC> idxBuf, metaBuf, codesBuf;
    TQue<QuePosition::VECOUT, 1> outQ;
    LocalTensor<float> sUb, cUb, vecUb, scrUb, redUb;
    LocalTensor<half> sHUb;
    LocalTensor<half> halfUb, uhUb;
    LocalTensor<float> skUb, svUb, tmpUb, tmp2Ub;
    LocalTensor<int64_t> idxUb;
    LocalTensor<int32_t> fillUb;
    LocalTensor<int8_t> codesUb;
    uint32_t myStart = 0, myCount = 0, batch = 1, hNum = 32, phase = 3;
    int64_t sSlot = 0;
};

extern "C" __global__ __aicore__ void gdn_lq_ftail(
    GM_ADDR state, GM_ADDR idx, GM_ADDR s_stage, GM_ADDR c_stage,
    GM_ADDR vq_stage, GM_ADDR kq_stage, GM_ADDR workspace, GM_ADDR tiling)
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
    GdnLqFtailKernel op;
    op.Init(state, idx, s_stage, c_stage, vq_stage, kq_stage, myStart,
            myCount, tilingData.numInstances / tilingData.h, tilingData.h,
            tilingData.sSlot, tilingData.phase, &pipe);
    op.LoadIdx();
    op.Process();
}
