#include "kernel_operator.h"

using namespace AscendC;

// pure vector-core kernel: keep the framework out of MIX codegen
ENABLE_FEATURE_FOR_COMPILE(default, KERNEL_TYPE_AIV_ONLY);

// LeapQuant GDN decode step for one (sequence, head) instance — the AscendC
// counterpart of the triton gdn_lq_step kernel / torch
// GDNLeapQuantPools.decode_step. The state is never materialized: the two
// state matvecs (Skn = S kn, Sqn = S qn*scale) are factorized through the
// composite layout, so the inner loops are vector FMAs / batched row dots
// instead of per-row scalar broadcasts:
//
//   S = pn*(codes*(sk/127)(x)sv + sum_r q_r(x)u_r) + sum_{j<f}(w_j at)ubuf_j(x)kbuf_j
//   Skn[v] = pn*sv[v]*(ck/127)*codesKn[v] + pn*sum_r cu_r*q_r[v]
//            + sum_{j<f} ckj_j*ubuf[j,v]     (cu_r=u_r.kn, ckj_j=(w_j at)(kbuf_j.kn))
//   Sqn[v] = pn*sv[v]*(cq/127)*codesQn[v] + pn*sum_r cq_r*q_r[v]
//            + sum_{j<f} cqk_j*ubuf[j,v]    (cq family with qn*scale)
//   u_new = beta*(v - Skn);  kq = kn.(qn*scale);  o = Sqn + kq*u_new
//
// Dead window rows (j >= f) may hold non-finite bf16 — they are skipped by a
// scalar branch, never multiplied by zero. fp32 accumulation throughout.
// Fixed geometry: V=K=128, R=4, L=16, H=32 layout (same as GdnLqFlush).

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr uint32_t F_R = 4;
constexpr uint32_t F_L = 16;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr float QMAX = 127.0f;

constexpr uint32_t OFF_CODES = 0;
constexpr uint32_t OFF_KBUF = 524288;
constexpr uint32_t OFF_UBUF = 655360;
constexpr uint32_t OFF_U = 786432;
constexpr uint32_t OFF_Q = 819200;
constexpr uint32_t OFF_SK = 852992;
constexpr uint32_t OFF_SV = 869376;
constexpr uint32_t OFF_W = 885760;
constexpr uint32_t OFF_P = 887808;
constexpr uint32_t OFF_FILL = 887936;

constexpr uint32_t HS_CODES = 16384;
constexpr uint32_t HS_KBUF = 4096;
constexpr uint32_t HS_UBUF = 4096;
constexpr uint32_t HS_UQ = 1024;
constexpr uint32_t HS_SKV = 512;
constexpr uint32_t HS_W = 64;

__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

// slot[0] = sum of src[0..count); src is clobbered. Proven helper from
// gdn_sketch_step; slot must be 32-byte aligned (never an element view).
// The compute mask is preset ONCE in Init (every reduce here uses
// count=128 -> mask 64); per-call SetMask would serialize the whole
// reduce pipeline through the shared mask register.
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
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
    WholeReduceSum(dst_local, src_local, ELEM_PER_REP_FP32, 1, 1, 1, 8);
#else
    WholeReduceSum<float, false>(dst_local, src_local, ELEM_PER_REP_FP32, 1, 1,
                                 1, 8);
#endif
}

class GdnLqStepKernel {
public:
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR idx, GM_ADDR kn,
                                GM_ADDR qn, GM_ADDR v, GM_ADDR gLog,
                                GM_ADDR beta, GM_ADDR out,
                                uint32_t myStart, uint32_t myCount,
                                uint32_t t, uint32_t h,
                                int64_t sSlot, float scale, TPipe* pipe)
    {

        this->myStart = myStart;
        this->myCount = myCount;
        this->batch = t;
        this->hNum = h;
        this->sSlot = sSlot;
        this->scale = scale;

        stateRaw = reinterpret_cast<__gm__ uint8_t*>(state);
        gmOutBase = reinterpret_cast<__gm__ bfloat16_t*>(out);
        gmIdx.SetGlobalBuffer((__gm__ int64_t*)idx);
        gmKn.SetGlobalBuffer((__gm__ bfloat16_t*)kn);
        gmQn.SetGlobalBuffer((__gm__ bfloat16_t*)qn);
        gmV.SetGlobalBuffer((__gm__ bfloat16_t*)v);
        gmG.SetGlobalBuffer((__gm__ float*)gLog);
        gmBeta.SetGlobalBuffer((__gm__ bfloat16_t*)beta);

        auto align32 = [](uint32_t b) { return (b + 31) / 32 * 32; };
        pipe->InitBuffer(codesBuf, F_V * F_K * sizeof(int8_t));
        pipe->InitBuffer(codesHBuf, F_V * F_K * sizeof(half));
        pipe->InitBuffer(codesFBuf, F_V * F_K * sizeof(float));
        pipe->InitBuffer(kbufRaw, F_L * F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(ubufRaw, F_L * F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(uqRaw, 2 * F_R * F_V * sizeof(half));
        pipe->InitBuffer(kbufF, F_L * F_K * sizeof(float));
        pipe->InitBuffer(ubufF, F_L * F_K * sizeof(float));
        pipe->InitBuffer(uF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(qF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(skF, F_K * sizeof(float));
        pipe->InitBuffer(svF, F_V * sizeof(float));
        pipe->InitBuffer(kqvF, 3 * F_K * sizeof(float));   // kn, qn*scale, v
        pipe->InitBuffer(knRaw, align32(F_K * sizeof(bfloat16_t)));
        pipe->InitBuffer(qnRaw, align32(F_K * sizeof(bfloat16_t)));
        pipe->InitBuffer(vRaw, align32(F_V * sizeof(bfloat16_t)));
        pipe->InitBuffer(wF, align32(F_L * sizeof(float)));
        pipe->InitBuffer(metaBuf, 5 * 32);  // g, betaF, p, fill, betaRaw
        pipe->InitBuffer(scrBuf, 38 * F_K * sizeof(float));
        pipe->InitBuffer(redBuf, 256 * sizeof(float));  // 16 batches x 16 contiguous results
        pipe->InitBuffer(idxBuf, align32(t * sizeof(int64_t)));
        // largest staged payload: two bf16 rows (kbuf/ubuf append) or o row
        pipe->InitBuffer(outQ, 1, 4 * F_V * sizeof(bfloat16_t));

        codesUb = codesBuf.Get<int8_t>();
        codesHUb = codesHBuf.Get<half>();
        codesFUb = codesFBuf.Get<float>();
        kbufRawUb = kbufRaw.Get<bfloat16_t>();
        ubufRawUb = ubufRaw.Get<bfloat16_t>();
        uqRawUb = uqRaw.Get<half>();
        kbufFUb = kbufF.Get<float>();
        ubufFUb = ubufF.Get<float>();
        uFUb = uF.Get<float>();
        qFUb = qF.Get<float>();
        skFUb = skF.Get<float>();
        svFUb = svF.Get<float>();
        knFUb = kqvF.Get<float>();
        qnFUb = kqvF.Get<float>()[F_K];
        vFUb = kqvF.Get<float>()[2 * F_K];
        knRawUb = knRaw.Get<bfloat16_t>();
        qnRawUb = qnRaw.Get<bfloat16_t>();
        vRawUb = vRaw.Get<bfloat16_t>();
        wFUb = wF.Get<float>();
        // meta slots (32B each): [0] g, [1] beta, [2] p, [3] fill
        gUb = metaBuf.Get<float>();
        betaUb = metaBuf.Get<float>()[8];
        pUb = metaBuf.Get<float>()[16];
        betaRawUb = metaBuf.Get<bfloat16_t>()[16];  // slot 4 = float 32
        scrUb = scrBuf.Get<float>();
        redUb = redBuf.Get<float>();
        idxUb = idxBuf.Get<int64_t>();
        fillUb = metaBuf.Get<int32_t>()[24];

        DataCopyPadParams pad{false, 0, 0, 0};
        this->padParams = pad;
        DataCopyPad(idxUb, gmIdx[0],
                    DataCopyParams{1, static_cast<uint16_t>(batch * 8), 0, 0},
                    padParams);
        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }

private:
    // V->S: scalar (S-pipe) GetValue reads of V-computed values need a
    // manual V_S event (constant id, single-queue kernel — see GdnLqFlush).
    __aicore__ inline void SyncV2S()
    {
        SetFlag<HardEvent::V_S>(STEP_V_S_EVT);
        WaitFlag<HardEvent::V_S>(STEP_V_S_EVT);
    }

    // slot[0] = dot(a, b); tmp (>=128 floats) is clobbered
    __aicore__ inline void Dot(const LocalTensor<float>& slot,
                               const LocalTensor<float>& a,
                               const LocalTensor<float>& b,
                               const LocalTensor<float>& tmp)
    {
        Mul(tmp, a, b, F_K);
        ReduceSumHalfInterval(slot, tmp, F_K);
    }

    // one batch of 16 codes-row dots: dst[v0+i] = sum_k codes[v+i,k]*vec[k].
    // Each row is halved to 64 lanes (one elementwise Add), the 16 rows are
    // packed contiguously, and ONE repeatTime=16 WholeReduceSum produces all
    // 16 partial sums contiguously — the reduce port is occupied once per
    // batch instead of 16 times (a lone WRS costs ~50+ cycles and does not
    // overlap with its neighbours).
    __aicore__ inline void CodesDots16(const LocalTensor<float>& dst,
                                       uint32_t v0,
                                       const LocalTensor<float>& vec)
    {
        // rows stay at 128-float stride; two repeat=16 WRS (low/high 64
        // halves) + one 16-wide add produce the 16 row sums contiguously
        uint32_t base = (v0 / 16) * 16;
        for (uint32_t i = 0; i < 16; ++i) {
            LocalTensor<float> row = scrUb[i * F_K];
            Mul(row, codesFUb[(v0 + i) * F_K], vec, F_K);
        }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
        WholeReduceSum(redUb[128], scrUb[0], ELEM_PER_REP_FP32, 16, 1, 1, 16);
        WholeReduceSum(redUb[192], scrUb[64], ELEM_PER_REP_FP32, 16, 1, 1, 16);
#else
        WholeReduceSum<float, false>(redUb[128], scrUb[0],
                                     ELEM_PER_REP_FP32, 16, 1, 1, 16);
        WholeReduceSum<float, false>(redUb[192], scrUb[64],
                                     ELEM_PER_REP_FP32, 16, 1, 1, 16);
#endif
        Add(redUb[base], redUb[128], redUb[192], 16);
    }

#ifdef GDN_LQ_STEP_DEBUG
    __aicore__ inline void DumpDbg(__gm__ float* dst,
                                   const LocalTensor<float>& src)
    {
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer(dst);
        LocalTensor<float> q = outQ.AllocTensor<float>();
        DataCopy(q, src, F_K);
        outQ.EnQue(q);
        LocalTensor<float> d = outQ.DeQue<float>();
        DataCopy(gm[0], d, F_K);
        outQ.FreeTensor(d);
    }
#endif

    __aicore__ inline void ProcessOne(uint32_t inst)
    {
        uint32_t b = inst / hNum;
        uint32_t h = inst % hNum;
        int32_t slot = idxUb.ReinterpretCast<int32_t>().GetValue(b * 2);
        __gm__ uint8_t* base = stateRaw
            + static_cast<uint64_t>(slot) * static_cast<uint64_t>(sSlot);
        uint64_t bh = static_cast<uint64_t>(b) * hNum + h;

        GlobalTensor<int32_t> gmFill;
        gmFill.SetGlobalBuffer((__gm__ int32_t*)(base + OFF_FILL));
        GlobalTensor<float> gmP;
        gmP.SetGlobalBuffer((__gm__ float*)(base + OFF_P));
        GlobalTensor<int8_t> gmCodes;
        gmCodes.SetGlobalBuffer((__gm__ int8_t*)(base + OFF_CODES
                                                 + h * HS_CODES));
        GlobalTensor<bfloat16_t> gmKbuf;
        gmKbuf.SetGlobalBuffer((__gm__ bfloat16_t*)(base + OFF_KBUF
                                                     + h * HS_KBUF));
        GlobalTensor<bfloat16_t> gmUbuf;
        gmUbuf.SetGlobalBuffer((__gm__ bfloat16_t*)(base + OFF_UBUF
                                                     + h * HS_UBUF));
        GlobalTensor<half> gmU;
        gmU.SetGlobalBuffer((__gm__ half*)(base + OFF_U + h * HS_UQ));
        GlobalTensor<half> gmQ;
        gmQ.SetGlobalBuffer((__gm__ half*)(base + OFF_Q + h * HS_UQ));
        GlobalTensor<float> gmSk;
        gmSk.SetGlobalBuffer((__gm__ float*)(base + OFF_SK + h * HS_SKV));
        GlobalTensor<float> gmSv;
        gmSv.SetGlobalBuffer((__gm__ float*)(base + OFF_SV + h * HS_SKV));
        GlobalTensor<float> gmW;
        gmW.SetGlobalBuffer((__gm__ float*)(base + OFF_W + h * HS_W));

        // per-(b,h) step inputs and scalars
        DataCopy(codesUb, gmCodes[0], F_V * F_K);
        DataCopy(kbufRawUb, gmKbuf[0], F_L * F_K);
        DataCopy(ubufRawUb, gmUbuf[0], F_L * F_K);
        DataCopy(uqRawUb, gmU[0], F_R * F_V);
        DataCopy(uqRawUb[F_R * F_V], gmQ[0], F_R * F_V);
        DataCopy(skFUb, gmSk[0], F_K);
        DataCopy(svFUb, gmSv[0], F_V);
        DataCopy(wFUb, gmW[0], F_L);
        DataCopy(knRawUb, gmKn[bh * F_K], F_K);
        DataCopy(qnRawUb, gmQn[bh * F_K], F_K);
        DataCopy(vRawUb, gmV[bh * F_V], F_V);
        DataCopyPad(gUb, gmG[bh], DataCopyParams{1, 4, 0, 0}, padParams);
        DataCopyPad(betaRawUb, gmBeta[bh], DataCopyParams{1, 2, 0, 0},
                    padParams);
        DataCopyPad(pUb, gmP[h], DataCopyParams{1, 4, 0, 0}, padParams);
        DataCopyPad(fillUb, gmFill[h], DataCopyParams{1, 4, 0, 0}, padParams);
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

        // ---- bulk casts ----
        Cast(kbufFUb, kbufRawUb, RoundMode::CAST_NONE, F_L * F_K);
        Cast(ubufFUb, ubufRawUb, RoundMode::CAST_NONE, F_L * F_K);
        Cast(uFUb, uqRawUb, RoundMode::CAST_NONE, F_R * F_V);
        Cast(qFUb, uqRawUb[F_R * F_V], RoundMode::CAST_NONE, F_R * F_V);
        Cast(knFUb, knRawUb, RoundMode::CAST_NONE, F_K);
        Cast(qnFUb, qnRawUb, RoundMode::CAST_NONE, F_K);
        Cast(vFUb, vRawUb, RoundMode::CAST_NONE, F_V);
        Muls(qnFUb, qnFUb, scale, F_K);  // qn_s = qn * scale

        // ---- per-instance contracted scalars, 8 dots per V_S round ----
        // codes rows are dotted with sk*(kn|qn_s)/127 (sk stays inside the
        // k-sum — it cannot be factored into a scalar ck)
        LocalTensor<float> kn2 = scrUb[18 * F_K];
        LocalTensor<float> qn2 = scrUb[19 * F_K];
        Mul(kn2, skFUb, knFUb, F_K);
        Muls(kn2, kn2, 1.0f / QMAX, F_K);
        Mul(qn2, skFUb, qnFUb, F_K);
        Muls(qn2, qn2, 1.0f / QMAX, F_K);
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
        // window rows, live only: ckj_j = (w_j*at)*(kbuf_j.kn),
        // cqk_j = (w_j*at)*(kbuf_j.qn_s); dead rows are skipped (non-finite
        // stale bytes must never be multiplied by zero)
        for (uint32_t j0 = 0; j0 < F_L; j0 += 8) {
            uint32_t cnt = (nj > j0 + 8) ? 8 : ((nj > j0) ? nj - j0 : 0);
            for (uint32_t i = 0; i < cnt; ++i) {
                uint32_t j = j0 + i;
                Dot(redUb[i * 8], kbufFUb[j * F_K], knFUb, scrUb[i * F_K]);
                Dot(redUb[64 + i * 8], kbufFUb[j * F_K], qnFUb,
                    scrUb[8 * F_K]);
            }
            SyncV2S();
            for (uint32_t i = 0; i < cnt; ++i) {
                uint32_t j = j0 + i;
                float wj = wFUb.GetValue(j);  // MTE2-loaded: still valid
                float wat = wj * at;
                ckj[j] = wat * redUb[i * 8].GetValue(0);
                cqk[j] = wat * redUb[64 + i * 8].GetValue(0);
            }
        }

        // whole-tile int8 -> fp16 -> fp32 casts (two ops replace 512 row
        // casts; codes values are integers in [-127,127], both stages exact)
        Cast(codesHUb, codesUb, RoundMode::CAST_NONE, F_V * F_K);
        Cast(codesFUb, codesHUb, RoundMode::CAST_NONE, F_V * F_K);
        // ---- codes row dots: codesKn[v] = codes_row_v . kn,
        //      codesQn[v] = codes_row_v . qn_s (8 rows per V_S round) ----
        LocalTensor<float> codesKn = scrUb[16 * F_K];
        LocalTensor<float> codesQn = scrUb[17 * F_K];
        for (uint32_t v0 = 0; v0 < F_V; v0 += 16) {
            CodesDots16(codesKn, v0, kn2);
        }
        // batch results are contiguous in redUb[0:128] — one V->V copy
        // replaces 128 GetValue/SetValue round trips (no S-pipe, no sync)
        DataCopy(codesKn, redUb, F_V);
        for (uint32_t v0 = 0; v0 < F_V; v0 += 16) {
            CodesDots16(codesQn, v0, qn2);
        }
        DataCopy(codesQn, redUb, F_V);

        // ---- assemble Skn / Sqn / u_new / o (all [128] vector ops) ----
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
        // window sums (dead rows already excluded from ckj/cqk); two
        // independent accumulator chains for ILP
        Duplicate(accA, 0.0f, F_K);
        Duplicate(accB, 0.0f, F_K);
        uint32_t nj2 = nj / 2;
        for (uint32_t j = 0; j < nj2; ++j) {
            Muls(tmpRow, ubufFUb[j * F_K], ckj[j], F_K);
            Add(accA, accA, tmpRow, F_K);
        }
        for (uint32_t j = nj2; j < nj; ++j) {
            Muls(tmpRow, ubufFUb[j * F_K], ckj[j], F_K);
            Add(accB, accB, tmpRow, F_K);
        }
        Add(accA, accA, accB, F_K);
        Add(skn, skn, accA, F_K);
        // u_new = beta * (v - Skn)
        Sub(unew, vFUb, skn, F_K);
        Muls(unew, unew, beta, F_K);
        Duplicate(accA, 0.0f, F_K);
        Duplicate(accB, 0.0f, F_K);
        for (uint32_t j = 0; j < nj2; ++j) {
            Muls(tmpRow, ubufFUb[j * F_K], cqk[j], F_K);
            Add(accA, accA, tmpRow, F_K);
        }
        for (uint32_t j = nj2; j < nj; ++j) {
            Muls(tmpRow, ubufFUb[j * F_K], cqk[j], F_K);
            Add(accB, accB, tmpRow, F_K);
        }
        Add(accA, accA, accB, F_K);
        Add(sqn, sqn, accA, F_K);
        // o = Sqn + kq * u_new
        Muls(tmpRow, unew, kq, F_K);
        Add(sqn, sqn, tmpRow, F_K);

#ifdef GDN_LQ_STEP_DEBUG
        if (inst == 0) {
            // slot 0 padding area (past the 888068-byte layout)
            __gm__ float* dbg =
                reinterpret_cast<__gm__ float*>(stateRaw + 1500000);
            DumpDbg(dbg + 0 * F_K, kn2);
            DumpDbg(dbg + 1 * F_K, qn2);
            DumpDbg(dbg + 2 * F_K, codesKn);
            DumpDbg(dbg + 3 * F_K, codesQn);
            DumpDbg(dbg + 4 * F_K, svFUb);
            DumpDbg(dbg + 5 * F_K, skn);
            DumpDbg(dbg + 6 * F_K, sqn);
            DumpDbg(dbg + 7 * F_K, unew);
        }
#endif
        // ---- stores: stage every payload into ONE queue buffer so the
        // framework V->MTE3 ordering is paid once per instance, then fan
        // out with plain DataCopys from the dequeued tensor.
        // Payload layout (1KB buffer):
        //   bf16 [0:128]   o row
        //   bf16 [128:256] kbuf[f] row (raw kn passthrough)
        //   bf16 [256:384] ubuf[f] row (u_new)
        //   f32  [192:208] w (entry f set to 1.0)
        //   f32  [208:216] p = pn
        //   f32  [224:232] fill = f+1 (float staging)
        //   i32  [232:240] fill (CAST_RINT result)
        LocalTensor<bfloat16_t> oq = outQ.AllocTensor<bfloat16_t>();
        Cast(oq, sqn, RoundMode::CAST_RINT, F_V);
        DataCopy(oq[F_V], knRawUb, F_K);
        Cast(oq[2 * F_V], unew, RoundMode::CAST_RINT, F_V);
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
        gmO.SetGlobalBuffer(gmOutBase + bh * F_V);
        DataCopy(gmO[0], oqD, F_V);
        DataCopy(gmKbuf[(uint64_t)f * F_K], oqD[F_V], F_K);
        DataCopy(gmUbuf[(uint64_t)f * F_V], oqD[2 * F_V], F_V);
        LocalTensor<float> wqD = oqD.ReinterpretCast<float>()[192];
        DataCopy(gmW[0], wqD, F_L);
        DataCopyPad(gmP[h], wqD[16], DataCopyParams{1, 4, 0, 0});
        DataCopyPad(gmFill[h], oqD.ReinterpretCast<int32_t>()[232],
                    DataCopyParams{1, 4, 0, 0});
        outQ.FreeTensor(oqD);
    }

    __gm__ bfloat16_t* gmOutBase = nullptr;
    GlobalTensor<int64_t> gmIdx;
    GlobalTensor<bfloat16_t> gmKn, gmQn, gmV;
    GlobalTensor<float> gmG;
    GlobalTensor<bfloat16_t> gmBeta;
    __gm__ uint8_t* stateRaw = nullptr;
    DataCopyPadParams padParams{false, 0, 0, 0};
    TBuf<QuePosition::VECCALC> codesBuf, codesHBuf, codesFBuf, kbufRaw;
    TBuf<QuePosition::VECCALC> ubufRaw, uqRaw;
    TBuf<QuePosition::VECCALC> kbufF, ubufF, uF, qF, skF, svF, kqvF, knRaw, qnRaw, vRaw, wF;
    TBuf<QuePosition::VECCALC> metaBuf, scrBuf, redBuf, idxBuf;
    TQue<QuePosition::VECOUT, 1> outQ;
    LocalTensor<int8_t> codesUb;
    LocalTensor<half> codesHUb;
    LocalTensor<float> codesFUb;
    LocalTensor<half> hstageUb;
    LocalTensor<bfloat16_t> knRawUb, qnRawUb, vRawUb, betaRawUb;
    LocalTensor<bfloat16_t> kbufRawUb, ubufRawUb;
    LocalTensor<half> uqRawUb;
    LocalTensor<float> kbufFUb, ubufFUb, uFUb, qFUb, skFUb, svFUb;
    LocalTensor<float> knFUb, qnFUb, vFUb, wFUb, gUb, betaUb, pUb;
    LocalTensor<float> scrUb, redUb;
    LocalTensor<int64_t> idxUb;
    LocalTensor<int32_t> fillUb;
    static constexpr uint32_t STEP_V_S_EVT = 2;
    uint32_t myStart = 0, myCount = 0, batch = 0, hNum = 32;
    int64_t sSlot = 0;
    float scale = 1.0f;
};

extern "C" __global__ __aicore__ void gdn_lq_step(
    GM_ADDR state, GM_ADDR idx, GM_ADDR kn, GM_ADDR qn, GM_ADDR v,
    GM_ADDR gLog, GM_ADDR beta, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{

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
    GdnLqStepKernel op;
    op.Init(state, idx, kn, qn, v, gLog, beta, out, myStart, myCount,
            tilingData.batch, tilingData.h, tilingData.sSlot,
            tilingData.scale, &pipe);
    op.Process();
}
