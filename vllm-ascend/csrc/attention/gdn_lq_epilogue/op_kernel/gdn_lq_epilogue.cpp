#include "kernel_operator.h"

using namespace AscendC;

// M2-D2 step v2, stage 3 (AIV-only): the v1 GdnLqStep assemble half with
// the codes row-dots replaced by the stage-2 cube output. Identical math
// and in-place window append contract as gdn_lq_step (fp32 accumulation,
// dead window rows skipped by scalar branch, never multiplied by zero);
// the 256 WholeReduceSum per head that v1 spent in CodesDots16 move to the
// cube. Fixed geometry V=K=128, R=4, L=16, composite per-head 32 KB block.

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr uint32_t F_R = 4;
constexpr uint32_t F_L = 16;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;

constexpr uint32_t HEAD_BYTES = 32768;
constexpr uint32_t OFF_SV = 16896;
constexpr uint32_t OFF_U = 17408;
constexpr uint32_t OFF_Q = 18432;
constexpr uint32_t OFF_KBUF = 19456;
constexpr uint32_t OFF_UBUF = 23552;
constexpr uint32_t OFF_W = 27648;
constexpr uint32_t OFF_P = 27712;
constexpr uint32_t OFF_FILL = 27716;

// canonical A3 fp32 row-sum helper (from gdn_lq_step, proven there)
__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

// slot[0] = sum of src[0..count); src is clobbered. The mask MUST be set
// immediately before every WholeReduceSum: the persistent mask register is
// clobbered by ordinary vector ops (gdn_lq_step finding).
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

class GdnLqEpilogueKernel {
public:
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR idx, GM_ADDR cq,
                                GM_ADDR kn, GM_ADDR qn, GM_ADDR v,
                                GM_ADDR gLog, GM_ADDR beta, GM_ADDR out,
                                uint32_t myStart, uint32_t myCount,
                                uint32_t batchNum, uint32_t hNum,
                                uint32_t nkNum, int64_t sSlot, float scale,
                                TPipe* pipe)
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
        cqBase = reinterpret_cast<__gm__ float*>(cq);
        knBase = reinterpret_cast<__gm__ bfloat16_t*>(kn);
        qnBase = reinterpret_cast<__gm__ bfloat16_t*>(qn);
        vBase = reinterpret_cast<__gm__ bfloat16_t*>(v);
        gBase = reinterpret_cast<__gm__ float*>(gLog);
        betaBase = reinterpret_cast<__gm__ bfloat16_t*>(beta);
        gmOutBase = reinterpret_cast<__gm__ bfloat16_t*>(out);

        pipe->InitBuffer(kbufRaw, F_L * F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(ubufRaw, F_L * F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(uqRaw, 2 * F_R * F_V * sizeof(half));
        pipe->InitBuffer(kbufF, F_L * F_K * sizeof(float));
        pipe->InitBuffer(ubufF, F_L * F_K * sizeof(float));
        pipe->InitBuffer(uF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(qF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(svF, F_V * sizeof(float));
        pipe->InitBuffer(kqvF, 3 * F_K * sizeof(float));   // kn, qn*scale, v
        pipe->InitBuffer(knRaw, F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(qnRaw, F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(vRaw, F_V * sizeof(bfloat16_t));
        pipe->InitBuffer(wF, F_L * sizeof(float));
        pipe->InitBuffer(metaBuf, 5 * 32);  // g, betaF, p, fill, betaRaw
        pipe->InitBuffer(cqBuf, 2 * F_V * sizeof(float));
        pipe->InitBuffer(scrBuf, 38 * F_K * sizeof(float));
        pipe->InitBuffer(redBuf, 2048 * sizeof(float));
        pipe->InitBuffer(idxBuf, batchNum * sizeof(int64_t));
        pipe->InitBuffer(outQ, 1, 4 * F_V * sizeof(bfloat16_t));

        kbufRawUb = kbufRaw.Get<bfloat16_t>();
        ubufRawUb = ubufRaw.Get<bfloat16_t>();
        uqRawUb = uqRaw.Get<half>();
        kbufFUb = kbufF.Get<float>();
        ubufFUb = ubufF.Get<float>();
        uFUb = uF.Get<float>();
        qFUb = qF.Get<float>();
        svFUb = svF.Get<float>();
        knFUb = kqvF.Get<float>();
        qnFUb = kqvF.Get<float>()[F_K];
        vFUb = kqvF.Get<float>()[2 * F_K];
        knRawUb = knRaw.Get<bfloat16_t>();
        qnRawUb = qnRaw.Get<bfloat16_t>();
        vRawUb = vRaw.Get<bfloat16_t>();
        wFUb = wF.Get<float>();
        gUb = metaBuf.Get<float>();
        betaUb = metaBuf.Get<float>()[8];
        pUb = metaBuf.Get<float>()[16];
        betaRawUb = metaBuf.Get<bfloat16_t>()[16];
        fillUb = metaBuf.Get<int32_t>()[24];
        cqUb = cqBuf.Get<float>();
        scrUb = scrBuf.Get<float>();
        redUb = redBuf.Get<float>();
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
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }

private:
    __aicore__ inline void SyncV2S()
    {
        SetFlag<HardEvent::V_S>(EPI_V_S_EVT);
        WaitFlag<HardEvent::V_S>(EPI_V_S_EVT);
    }

    __aicore__ inline void Dot(const LocalTensor<float>& slot,
                               const LocalTensor<float>& a,
                               const LocalTensor<float>& b,
                               const LocalTensor<float>& tmp)
    {
        Mul(tmp, a, b, F_K);
        ReduceSumHalfInterval(slot, tmp, F_K);
    }

    __aicore__ inline void ProcessOne(uint32_t inst)
    {
        uint32_t b = inst / hNum;
        uint32_t h = inst % hNum;
        uint32_t kh = h / rep;
        int32_t slot = idxUb.ReinterpretCast<int32_t>().GetValue(b * 2);
        __gm__ uint8_t* base = stateRaw
            + static_cast<uint64_t>(slot) * static_cast<uint64_t>(sSlot)
            + static_cast<uint64_t>(h) * HEAD_BYTES;

        GlobalTensor<float> gmSv;
        gmSv.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + OFF_SV));
        GlobalTensor<half> gmU;
        gmU.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(base + OFF_U));
        GlobalTensor<half> gmQ;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(base + OFF_Q));
        GlobalTensor<bfloat16_t> gmKbuf;
        gmKbuf.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(base + OFF_KBUF));
        GlobalTensor<bfloat16_t> gmUbuf;
        gmUbuf.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(base + OFF_UBUF));
        GlobalTensor<float> gmW;
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + OFF_W));
        GlobalTensor<float> gmP;
        gmP.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base + OFF_P));
        GlobalTensor<int32_t> gmFill;
        gmFill.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(base + OFF_FILL));

        GlobalTensor<float> gmCq0;
        gmCq0.SetGlobalBuffer(cqBase
                              + (static_cast<uint64_t>(inst) * 16 + 0) * F_V);
        GlobalTensor<float> gmCq1;
        gmCq1.SetGlobalBuffer(cqBase
                              + (static_cast<uint64_t>(inst) * 16 + 1) * F_V);
        GlobalTensor<bfloat16_t> gmKnRow;
        gmKnRow.SetGlobalBuffer(knBase
            + (static_cast<uint64_t>(b) * (hNum / rep) + kh) * F_K);
        GlobalTensor<bfloat16_t> gmQnRow;
        gmQnRow.SetGlobalBuffer(qnBase
            + (static_cast<uint64_t>(b) * (hNum / rep) + kh) * F_K);
        GlobalTensor<bfloat16_t> gmVRow;
        gmVRow.SetGlobalBuffer(vBase + static_cast<uint64_t>(inst) * F_V);
        GlobalTensor<float> gmG;
        gmG.SetGlobalBuffer(gBase + inst);
        GlobalTensor<bfloat16_t> gmBeta;
        gmBeta.SetGlobalBuffer(betaBase + inst);

        DataCopy(kbufRawUb, gmKbuf[0], F_L * F_K);
        DataCopy(ubufRawUb, gmUbuf[0], F_L * F_K);
        DataCopy(uqRawUb, gmU[0], F_R * F_V);
        DataCopy(uqRawUb[F_R * F_V], gmQ[0], F_R * F_V);
        DataCopy(svFUb, gmSv[0], F_V);
        DataCopy(wFUb, gmW[0], F_L);
        // one codes-dot row = 128 fp32 = 512 B (DataCopyPad blockLen is
        // BYTES, cf. the 4-byte scalar pads above)
        DataCopyPad(cqUb, gmCq0[0], DataCopyParams{1, 4 * F_V, 0, 0},
                    padParams);
        DataCopyPad(cqUb[F_V], gmCq1[0], DataCopyParams{1, 4 * F_V, 0, 0},
                    padParams);
        DataCopy(knRawUb, gmKnRow[0], F_K);
        DataCopy(qnRawUb, gmQnRow[0], F_K);
        DataCopy(vRawUb, gmVRow[0], F_V);
        DataCopyPad(gUb, gmG[0], DataCopyParams{1, 4, 0, 0}, padParams);
        DataCopyPad(betaRawUb, gmBeta[0], DataCopyParams{1, 2, 0, 0},
                    padParams);
        DataCopyPad(pUb, gmP[0], DataCopyParams{1, 4, 0, 0}, padParams);
        DataCopyPad(fillUb, gmFill[0], DataCopyParams{1, 4, 0, 0}, padParams);
        PipeBarrier<PIPE_ALL>();

        // ---- scalar prep: at = exp(g); beta/p/fill reads ----
        AscendC::Exp(gUb, gUb, 8);  // lane 0 = exp(g)
        PipeBarrier<PIPE_V>();
        Cast(betaUb, betaRawUb, RoundMode::CAST_NONE, 8);
        PipeBarrier<PIPE_V>();
        SyncV2S();
        float at = gUb.GetValue(0);
        float beta = betaUb.GetValue(0);
        float pval = pUb.GetValue(0);
        int32_t f = fillUb.GetValue(0);
        float pn = pval * at;
        uint32_t nj = (f < static_cast<int32_t>(F_L))
                          ? static_cast<uint32_t>(f)
                          : static_cast<uint32_t>(F_L);

        for (uint32_t j = 0; j < F_L; ++j) {
            Cast(kbufFUb[j * F_K], kbufRawUb[j * F_K], RoundMode::CAST_NONE,
                 F_K);
            Cast(ubufFUb[j * F_K], ubufRawUb[j * F_K], RoundMode::CAST_NONE,
                 F_K);
        }
        Cast(uFUb, uqRawUb, RoundMode::CAST_NONE, F_R * F_V);
        Cast(qFUb, uqRawUb[F_R * F_V], RoundMode::CAST_NONE, F_R * F_V);
        Cast(knFUb, knRawUb, RoundMode::CAST_NONE, F_K);
        Cast(qnFUb, qnRawUb, RoundMode::CAST_NONE, F_K);
        Cast(vFUb, vRawUb, RoundMode::CAST_NONE, F_V);
        Muls(qnFUb, qnFUb, scale, F_K);  // qn_s = qn * scale

        // ---- per-instance contracted scalars, 8 dots per V_S round ----
        float kq;
        float cu[F_R], cq_r[F_R], ckj[F_L], cqk[F_L];
        Dot(redUb[16], knFUb, qnFUb, scrUb[2 * F_K]);
        Dot(redUb[24], uFUb[0], knFUb, scrUb[3 * F_K]);
        Dot(redUb[32], uFUb[F_K], knFUb, scrUb[4 * F_K]);
        Dot(redUb[40], uFUb[2 * F_K], knFUb, scrUb[5 * F_K]);
        Dot(redUb[48], uFUb[3 * F_K], knFUb, scrUb[6 * F_K]);
        Dot(redUb[56], uFUb[0], qnFUb, scrUb[7 * F_K]);
        SyncV2S();
        kq = redUb[16].GetValue(0);
        cu[0] = redUb[24].GetValue(0);
        cu[1] = redUb[32].GetValue(0);
        cu[2] = redUb[40].GetValue(0);
        cu[3] = redUb[48].GetValue(0);
        cq_r[0] = redUb[56].GetValue(0);
        for (uint32_t r = 1; r < F_R; ++r) {
            Dot(redUb[(r - 1) * 8], uFUb[r * F_K], qnFUb,
                scrUb[(r - 1) * F_K]);
        }
        SyncV2S();
        for (uint32_t r = 1; r < F_R; ++r) {
            cq_r[r] = redUb[(r - 1) * 8].GetValue(0);
        }
        // live window rows only: ckj_j = kbuf_j.kn, cqk_j = kbuf_j.qn_s;
        // dead rows skipped (non-finite stale bytes never multiplied).
        // Fully unrolled with nj guards so coefficient arrays register-
        // promote (runtime-indexed local arrays miscompile on this toolchain).
        for (uint32_t j = 0; j < F_L; ++j) {
            if (j < nj) {
                Dot(redUb[j * 8], kbufFUb[j * F_K], knFUb,
                    scrUb[j * F_K]);
                Dot(redUb[128 + j * 8], kbufFUb[j * F_K], qnFUb,
                    scrUb[((j + 8) % 16) * F_K]);
            }
        }
        SyncV2S();
        for (uint32_t j = 0; j < F_L; ++j) {
            if (j < nj) {
                float wj = wFUb.GetValue(j);
                float wat = wj * at;
                ckj[j] = wat * redUb[j * 8].GetValue(0);
                cqk[j] = wat * redUb[128 + j * 8].GetValue(0);
            }
        }

        // ---- assemble Skn / Sqn / u_new / o (all [128] vector ops) ----
        // codesKn/codesQn are the stage-2 cube rows (s_k, 1/127 and scale
        // already folded into the k'/q' premultiplies of stage 1)
        LocalTensor<float> codesKn = cqUb;
        LocalTensor<float> codesQn = cqUb[F_V];
        LocalTensor<float> skn = scrUb[20 * F_K];
        LocalTensor<float> sqn = scrUb[21 * F_K];
        LocalTensor<float> unew = scrUb[22 * F_K];
        LocalTensor<float> accA = scrUb[23 * F_K];
        LocalTensor<float> accB = scrUb[24 * F_K];
        LocalTensor<float> tmpRow = scrUb[25 * F_K];
        Mul(skn, svFUb, codesKn, F_K);
        Muls(skn, skn, pn, F_K);
        Mul(sqn, svFUb, codesQn, F_K);
        Muls(sqn, sqn, pn, F_K);
        Duplicate(accA, 0.0f, F_K);
        Duplicate(accB, 0.0f, F_K);
        for (uint32_t r = 0; r < F_R; ++r) {
            Muls(tmpRow, qFUb[r * F_K], cu[r], F_K);
            Add(accA, accA, tmpRow, F_K);
            Muls(tmpRow, qFUb[r * F_K], cq_r[r], F_K);
            Add(accB, accB, tmpRow, F_K);
        }
        Muls(accA, accA, pn, F_K);
        Muls(accB, accB, pn, F_K);
        Add(skn, skn, accA, F_K);
        Add(sqn, sqn, accB, F_K);
        Duplicate(accA, 0.0f, F_K);
        Duplicate(accB, 0.0f, F_K);
        for (uint32_t j = 0; j < F_L; ++j) {
            if (j < nj) {
                Muls(tmpRow, ubufFUb[j * F_K], ckj[j], F_K);
                Add((j & 1) ? accB : accA, (j & 1) ? accB : accA, tmpRow,
                    F_K);
            }
        }
        Add(accA, accA, accB, F_K);
        Add(skn, skn, accA, F_K);
        // u_new = beta * (v - Skn)
        Sub(unew, vFUb, skn, F_K);
        Muls(unew, unew, beta, F_K);
        Duplicate(accA, 0.0f, F_K);
        Duplicate(accB, 0.0f, F_K);
        for (uint32_t j = 0; j < F_L; ++j) {
            if (j < nj) {
                Muls(tmpRow, ubufFUb[j * F_K], cqk[j], F_K);
                Add((j & 1) ? accB : accA, (j & 1) ? accB : accA, tmpRow,
                    F_K);
            }
        }
        Add(accA, accA, accB, F_K);
        Add(sqn, sqn, accA, F_K);
        // o = Sqn + kq * u_new
        Muls(tmpRow, unew, kq, F_K);
        Add(sqn, sqn, tmpRow, F_K);

        // ---- stores: same single-queue payload fan-out as gdn_lq_step
        //   bf16 [0:128]   o row
        //   bf16 [128:256] kbuf[f] row (raw kn passthrough)
        //   bf16 [256:384] ubuf[f] row (u_new)
        //   f32  [192:208] w (entry f = 1.0)
        //   f32  [208:216] p = pn
        //   i32  [232:240] fill = f+1
        LocalTensor<bfloat16_t> oq = outQ.AllocTensor<bfloat16_t>();
        Cast(oq, sqn, RoundMode::CAST_RINT, F_V);
        DataCopy(oq[F_V], knRawUb, F_K);
        Cast(oq[2 * F_V], unew, RoundMode::CAST_RINT, F_V);
        Muls(wFUb, wFUb, at, F_L);
        LocalTensor<float> wq = oq.ReinterpretCast<float>()[192];
        DataCopy(wq, wFUb, F_L);
        wq.SetValue(f, 1.0f);
        Duplicate(wq[16], pn, 8);
        Duplicate(wq[32], static_cast<float>(f + 1), 8);
        Cast(oq.ReinterpretCast<int32_t>()[232], wq[32],
             RoundMode::CAST_RINT, 8);
        outQ.EnQue(oq);
        LocalTensor<bfloat16_t> oqD = outQ.DeQue<bfloat16_t>();
        GlobalTensor<bfloat16_t> gmO;
        gmO.SetGlobalBuffer(gmOutBase + static_cast<uint64_t>(inst) * F_V);
        DataCopy(gmO[0], oqD, F_V);
        DataCopy(gmKbuf[(uint64_t)f * F_K], oqD[F_V], F_K);
        DataCopy(gmUbuf[(uint64_t)f * F_V], oqD[2 * F_V], F_V);
        LocalTensor<float> wqD = oqD.ReinterpretCast<float>()[192];
        DataCopy(gmW[0], wqD, F_L);
        DataCopyPad(gmP[0], wqD[16], DataCopyParams{1, 4, 0, 0});
        DataCopyPad(gmFill[0], oqD.ReinterpretCast<int32_t>()[232],
                    DataCopyParams{1, 4, 0, 0});
        outQ.FreeTensor(oqD);
    }

    static constexpr uint32_t EPI_V_S_EVT = 2;
    __gm__ uint8_t* stateRaw = nullptr;
    GlobalTensor<int64_t> gmIdx;
    __gm__ float* cqBase = nullptr;
    __gm__ bfloat16_t* knBase = nullptr;
    __gm__ bfloat16_t* qnBase = nullptr;
    __gm__ bfloat16_t* vBase = nullptr;
    __gm__ float* gBase = nullptr;
    __gm__ bfloat16_t* betaBase = nullptr;
    __gm__ bfloat16_t* gmOutBase = nullptr;
    DataCopyPadParams padParams{false, 0, 0, 0};
    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> kbufRaw, ubufRaw, uqRaw;
    TBuf<QuePosition::VECCALC> kbufF, ubufF, uF, qF, svF, kqvF;
    TBuf<QuePosition::VECCALC> knRaw, qnRaw, vRaw, wF;
    TBuf<QuePosition::VECCALC> metaBuf, cqBuf, scrBuf, redBuf, idxBuf;
    TQue<QuePosition::VECOUT, 1> outQ;
    LocalTensor<bfloat16_t> kbufRawUb, ubufRawUb;
    LocalTensor<half> uqRawUb;
    LocalTensor<float> kbufFUb, ubufFUb, uFUb, qFUb, svFUb;
    LocalTensor<float> knFUb, qnFUb, vFUb, wFUb;
    LocalTensor<float> gUb, betaUb, pUb;
    LocalTensor<bfloat16_t> knRawUb, qnRawUb, vRawUb, betaRawUb;
    LocalTensor<int32_t> fillUb;
    LocalTensor<float> cqUb;
    LocalTensor<float> scrUb, redUb;
    LocalTensor<int64_t> idxUb;
    uint32_t myStart = 0, myCount = 0, batch = 1, hNum = 32, rep = 2;
    int64_t sSlot = 0;
    float scale = 1.0f;
};

extern "C" __global__ __aicore__ void gdn_lq_epilogue(
    GM_ADDR state, GM_ADDR idx, GM_ADDR cq, GM_ADDR kn, GM_ADDR qn,
    GM_ADDR v, GM_ADDR gLog, GM_ADDR beta, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
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
    GdnLqEpilogueKernel op;
    op.Init(state, idx, cq, kn, qn, v, gLog, beta, out, myStart, myCount,
            tilingData.numInstances / tilingData.h, tilingData.h,
            tilingData.nk, tilingData.sSlot, tilingData.scale, &pipe);
    op.LoadIdx();
    op.Process();
}
