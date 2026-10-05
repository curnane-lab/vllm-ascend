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

// per-(slot, head) contiguous 32 KB block layout; head base = h*HEAD_BYTES
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
// The mask MUST be set immediately before every WholeReduceSum: the
// persistent mask register is clobbered by ordinary vector ops, and an
// isSetMask=false reduce silently uses whatever mask is left behind.
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
        pipe->InitBuffer(redBuf, 2048 * sizeof(float));  // 16 batches x 16 stride-8 slots
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
    // Each row is halved to 64 lanes (one elementwise Add), then ONE
    // repeatTime=1 WholeReduceSum per row writes the sum at
    // red[base + i*8] element 0 (the proven single-repeat form; the
    // repeat-form dstRepStride unit turned out to be datablock-dependent
    // and mis-scattered the sums). All 16 reduces queue on the reduce port
    // back-to-back; one V_S sync covers the whole batch.
    __aicore__ inline void CodesDots16(const LocalTensor<float>& dst,
                                       uint32_t v0,
                                       const LocalTensor<float>& vec)
    {
        LocalTensor<float> packed = scrUb[26 * F_K];  // 16 x 64
        uint32_t base = (v0 / 16) * 128;
        for (uint32_t i = 0; i < 16; ++i) {
            LocalTensor<float> row = scrUb[i * F_K];
            Mul(row, codesFUb[(v0 + i) * F_K], vec, F_K);
            Add(packed[i * 64], row, row[64], 64);
            // the Mul/Add above clobber the compute mask; the reduce with
            // isSetMask=false would silently eat whatever they left. Reset
            // per row (ReduceSumHalfInterval does the same).
            AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
            WholeReduceSum(redUb[base + i * 8], packed[i * 64],
                           ELEM_PER_REP_FP32, 1, 1, 1, 8);
#else
            WholeReduceSum<float, false>(redUb[base + i * 8], packed[i * 64],
                                         ELEM_PER_REP_FP32, 1, 1, 1, 8);
#endif
        }
        // NOTE(cce hazard): ckj/cqk below are local arrays indexed by a
        // runtime j; this build's compiler happens to fully unroll the
        // fixed-bound loop. If the loop is ever re-shaped so j is not a
        // compile-time constant, scalar reads can return garbage
        // (observed class of bug on this toolchain) — prefer indexed UB
        // slots or forced unroll then.
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
        gmFill.SetGlobalBuffer((__gm__ int32_t*)(base + OFF_FILL + h * HEAD_BYTES));
        GlobalTensor<float> gmP;
        gmP.SetGlobalBuffer((__gm__ float*)(base + OFF_P + h * HEAD_BYTES));
        GlobalTensor<int8_t> gmCodes;
        gmCodes.SetGlobalBuffer((__gm__ int8_t*)(base + OFF_CODES
                                                 + h * HEAD_BYTES));
        GlobalTensor<bfloat16_t> gmKbuf;
        gmKbuf.SetGlobalBuffer((__gm__ bfloat16_t*)(base + OFF_KBUF
                                                     + h * HEAD_BYTES));
        GlobalTensor<bfloat16_t> gmUbuf;
        gmUbuf.SetGlobalBuffer((__gm__ bfloat16_t*)(base + OFF_UBUF
                                                     + h * HEAD_BYTES));
        GlobalTensor<half> gmU;
        gmU.SetGlobalBuffer((__gm__ half*)(base + OFF_U + h * HEAD_BYTES));
        GlobalTensor<half> gmQ;
        gmQ.SetGlobalBuffer((__gm__ half*)(base + OFF_Q + h * HEAD_BYTES));
        GlobalTensor<float> gmSk;
        gmSk.SetGlobalBuffer((__gm__ float*)(base + OFF_SK + h * HEAD_BYTES));
        GlobalTensor<float> gmSv;
        gmSv.SetGlobalBuffer((__gm__ float*)(base + OFF_SV + h * HEAD_BYTES));
        GlobalTensor<float> gmW;
        gmW.SetGlobalBuffer((__gm__ float*)(base + OFF_W + h * HEAD_BYTES));

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

        // row-wise casts: a single call with count > 255 overflows the uint8
        // repeat field of the cast instruction and silently misbehaves
        // (observed: NaN-garbage codes rows on some heads). 128-element rows
        // stay well inside the limit.
        for (uint32_t v = 0; v < F_V; ++v) {
            Cast(codesHUb[v * F_K], codesUb[v * F_K], RoundMode::CAST_NONE, F_K);
            Cast(codesFUb[v * F_K], codesHUb[v * F_K], RoundMode::CAST_NONE, F_K);
        }
        for (uint32_t j = 0; j < F_L; ++j) {
            Cast(kbufFUb[j * F_K], kbufRawUb[j * F_K], RoundMode::CAST_NONE, F_K);
            Cast(ubufFUb[j * F_K], ubufRawUb[j * F_K], RoundMode::CAST_NONE, F_K);
        }
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
        // stale bytes must never be multiplied by zero). The loops are
        // fully unrolled over the compile-time j so the coefficient arrays
        // are register-promoted — runtime-indexed local arrays are
        // miscompiled by the cce frontend (observed: entries beyond index 0
        // read back as garbage, silently dropping window rows >= 1).
        for (uint32_t j = 0; j < F_L; ++j) {
            if (j < nj) {
                Dot(redUb[j * 8], kbufFUb[j * F_K], knFUb, scrUb[j * F_K]);
                Dot(redUb[128 + j * 8], kbufFUb[j * F_K], qnFUb,
                    scrUb[((j + 8) % 16) * F_K]);
            }
        }
        SyncV2S();
        for (uint32_t j = 0; j < F_L; ++j) {
            if (j < nj) {
                float wj = wFUb.GetValue(j);  // MTE2-loaded: still valid
                float wat = wj * at;
                ckj[j] = wat * redUb[j * 8].GetValue(0);
                cqk[j] = wat * redUb[128 + j * 8].GetValue(0);
            }
        }

        // ---- codes row dots: codesKn[v] = codes_row_v . kn,
        //      codesQn[v] = codes_row_v . qn_s (8 rows per V_S round) ----
        LocalTensor<float> codesKn = scrUb[16 * F_K];
        LocalTensor<float> codesQn = scrUb[17 * F_K];
        for (uint32_t v0 = 0; v0 < F_V; v0 += 16) {
            CodesDots16(codesKn, v0, kn2);
        }
        SyncV2S();
        for (uint32_t v = 0; v < F_V; ++v) {
            codesKn.SetValue(v, redUb[(v / 16) * 128 + (v % 16) * 8].GetValue(0));
        }
        PipeBarrier<PIPE_ALL>();  // S scatter -> next side's V reduces
        for (uint32_t v0 = 0; v0 < F_V; v0 += 16) {
            CodesDots16(codesQn, v0, qn2);
        }
        SyncV2S();
        for (uint32_t v = 0; v < F_V; ++v) {
            codesQn.SetValue(v, redUb[(v / 16) * 128 + (v % 16) * 8].GetValue(0));
        }
        PipeBarrier<PIPE_ALL>();

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
        // window sums (dead rows already excluded from ckj/cqk); fully
        // unrolled with nj guards (register-promoted coefficient arrays)
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
        // decay the live window weights, then stage w with entry f = 1.0
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
        gmO.SetGlobalBuffer(gmOutBase + bh * F_V);
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
