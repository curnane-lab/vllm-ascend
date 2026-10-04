#include "kernel_operator.h"


using namespace AscendC;

// pure vector-core kernel: keep the framework out of MIX codegen
ENABLE_FEATURE_FOR_COMPILE(default, KERNEL_TYPE_AIV_ONLY);

// Full SketchSSM decode step for one (row, value-head) instance — the
// AscendC counterpart of _gdn_sketch_step_k1/k2 fused into a single kernel.
// One AI vector core processes many instances sequentially; all reductions
// use the canonical A3 halving-sum helper (count <= 64).
// Fixed geometry: W=16, K=128, V=128.

constexpr uint32_t STEP_W = 16;
constexpr uint32_t STEP_K = 128;
constexpr uint32_t STEP_V = 128;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr uint32_t STEP_P = 4;

__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

// dst = sum of src[0..count); src is clobbered.
__aicore__ inline void ReduceSumHalfInterval(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
    int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / 2;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
    WholeReduceSum(dst_local, src_local, ELEM_PER_REP_FP32, 1, 1, 1,
                   ELEM_PER_BLK_FP32);
#else
    WholeReduceSum<float, false>(dst_local, src_local, ELEM_PER_REP_FP32, 1, 1,
                                 1, 8);
#endif
    PipeBarrier<PIPE_V>();
}

class GdnStepKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR mixedQkv, GM_ADDR aAct, GM_ADDR bAct, GM_ADDR aLog,
        GM_ADDR dtBias, GM_ADDR state, GM_ADDR dCache, GM_ADDR kCache,
        GM_ADDR gCache, GM_ADDR slots, GM_ADDR writePos, GM_ADDR meta,
        GM_ADDR u, GM_ADDR phi, GM_ADDR fs, GM_ADDR betaRing,
        GM_ADDR currentD, GM_ADDR currentK, GM_ADDR ranks, GM_ADDR layout,
        GM_ADDR out, uint32_t myStart, uint32_t myCount, uint32_t hv,
        uint32_t h, float scale, uint32_t mode, int64_t sQkv,
        int64_t sA, int64_t sB,
        int64_t sStSlot, int64_t sStHead, int64_t sDcSlot, int64_t sDcHead,
        int64_t sKcSlot, int64_t sKcHead, int64_t sGcSlot, int64_t sGcHead,
        int64_t sSu, int64_t sSm, int64_t sSf, TPipe* pipe)
    {
        this->myStart = myStart;
        this->myCount = myCount;
        this->hvNum = hv;
        this->hNum = h;
        this->hvPerH = hv / h;
        this->scale = scale;
        this->mode = mode;
        this->sQkv = sQkv;
        this->sA = sA;
        this->sB = sB;
        this->sStSlot = sStSlot;
        this->sStHead = sStHead;
        this->sDcSlot = sDcSlot;
        this->sDcHead = sDcHead;
        this->sKcSlot = sKcSlot;
        this->sKcHead = sKcHead;
        this->sGcSlot = sGcSlot;
        this->sGcHead = sGcHead;
        this->sSu = sSu;
        this->sSm = sSm;
        this->sSf = sSf;

        gmQkv.SetGlobalBuffer((__gm__ bfloat16_t*)mixedQkv);
        gmA.SetGlobalBuffer((__gm__ float*)aAct);
        gmB.SetGlobalBuffer((__gm__ float*)bAct);
        gmALog.SetGlobalBuffer((__gm__ float*)aLog);
        gmDtBias.SetGlobalBuffer((__gm__ float*)dtBias);
        gmState.SetGlobalBuffer((__gm__ float*)state);
        gmDCache.SetGlobalBuffer((__gm__ bfloat16_t*)dCache);
        gmKCache.SetGlobalBuffer((__gm__ bfloat16_t*)kCache);
        gmGCache.SetGlobalBuffer((__gm__ float*)gCache);
        gmSlots.SetGlobalBuffer((__gm__ int32_t*)slots);
        gmWritePos.SetGlobalBuffer((__gm__ int32_t*)writePos);
        gmMeta.SetGlobalBuffer((__gm__ int32_t*)meta);
        gmU.SetGlobalBuffer((__gm__ bfloat16_t*)u);
        gmPhi.SetGlobalBuffer((__gm__ bfloat16_t*)phi);
        gmFs.SetGlobalBuffer((__gm__ bfloat16_t*)fs);
        gmBetaRing.SetGlobalBuffer((__gm__ float*)betaRing);
        gmCurrentD.SetGlobalBuffer((__gm__ float*)currentD);
        gmCurrentK.SetGlobalBuffer((__gm__ float*)currentK);
        gmRanks.SetGlobalBuffer((__gm__ int32_t*)ranks);
        gmLayout.SetGlobalBuffer((__gm__ int32_t*)layout);
        gmOut.SetGlobalBuffer((__gm__ bfloat16_t*)out);


        uint32_t rowsLo = myStart / hvNum;
        uint32_t rowsHi = (myStart + myCount - 1) / hvNum;
        uint32_t nRows = rowsHi - rowsLo + 1;
        uint32_t rowsTot = rowsHi + 1;          // rows 0..rowsHi
        uint32_t rowsPad = (rowsTot + 7) & ~7u; // DataCopy: 32B-multiple len
        this->rowsLo = rowsLo;

        // MTE requires 32B-aligned buffers: round every preload up
        auto align32 = [](uint32_t b) { return (b + 31) / 32 * 32; };
        pipe->InitBuffer(metaSlots, align32(rowsPad * sizeof(int32_t)));
        pipe->InitBuffer(metaWp, align32(rowsPad * 2 * sizeof(int32_t)));
        pipe->InitBuffer(metaCidx, align32(rowsPad * sizeof(int32_t)));
        pipe->InitBuffer(metaA, align32(nRows * hvNum * sizeof(float)));
        pipe->InitBuffer(metaB, align32(nRows * hvNum * sizeof(float)));
        pipe->InitBuffer(metaALog, align32(hvNum * sizeof(float)));
        // inQQkv doubles as the Init meta staging buffer: sized for the
        // largest of (q+k+v) and the int64 write_pos preload
        uint32_t qkvBuf = 3 * STEP_K * sizeof(bfloat16_t);
        if (align32(rowsTot * 8) > qkvBuf) qkvBuf = align32(rowsTot * 8);
        pipe->InitBuffer(inQQkv, 1, qkvBuf);
        pipe->InitBuffer(metaDt, align32(hvNum * sizeof(float)));
        pipe->InitBuffer(metaRanks, align32(hvNum * sizeof(int32_t)));
        pipe->InitBuffer(metaLayout, align32(hvNum * 4 * sizeof(int32_t)));

        slotsUb = metaSlots.Get<int32_t>();
        wpUb = metaWp.Get<int32_t>();
        cidxUb = metaCidx.Get<int32_t>();
        aUb = metaA.Get<float>();
        bUb = metaB.Get<float>();
        alogUb = metaALog.Get<float>();
        dtUb = metaDt.Get<float>();
        ranksUb = metaRanks.Get<int32_t>();
        layoutUb = metaLayout.Get<int32_t>();
        // exact-length GM reads (any batch) via pad into the staging queue,
        // then 32B-multiple UB->UB copies into the working TBufs
        DataCopyPadParams metaPad{false, 0, 0, 0};
        LocalTensor<int32_t> mq = inQQkv.AllocTensor<int32_t>();
        DataCopyPad(mq, gmSlots[0],
                    DataCopyParams{1, static_cast<uint16_t>(rowsTot * 4), 0, 0},
                    metaPad);
        inQQkv.EnQue(mq);
        LocalTensor<int32_t> mqIn = inQQkv.DeQue<int32_t>();
        DataCopy(slotsUb, mqIn, rowsPad);
        inQQkv.FreeTensor(mqIn);
        mq = inQQkv.AllocTensor<int32_t>();
        DataCopyPad(mq, gmMeta[0],
                    DataCopyParams{1, static_cast<uint16_t>(rowsTot * 4), 0, 0},
                    metaPad);
        inQQkv.EnQue(mq);
        mqIn = inQQkv.DeQue<int32_t>();
        DataCopy(cidxUb, mqIn, rowsPad);
        inQQkv.FreeTensor(mqIn);
        // write_pos is int64; the low 32-bit word of row n lives at word 2n
        mq = inQQkv.AllocTensor<int32_t>();
        DataCopyPad(mq, gmWritePos[0],
                    DataCopyParams{1, static_cast<uint16_t>(rowsTot * 8), 0, 0},
                    metaPad);
        inQQkv.EnQue(mq);
        mqIn = inQQkv.DeQue<int32_t>();
        DataCopy(wpUb, mqIn, rowsPad * 2);
        inQQkv.FreeTensor(mqIn);
        DataCopy(aUb, gmA[(uint64_t)rowsLo * hvNum], nRows * hvNum);
        DataCopy(bUb, gmB[(uint64_t)rowsLo * hvNum], nRows * hvNum);
        DataCopy(alogUb, gmALog[0], hvNum);
        DataCopy(dtUb, gmDtBias[0], hvNum);
        DataCopy(ranksUb, gmRanks[0], hvNum);
        DataCopy(layoutUb, gmLayout[0], hvNum * 4);
        PipeBarrier<PIPE_ALL>();

        pipe->InitBuffer(inQRingK, 1, STEP_W * STEP_K * sizeof(bfloat16_t));
        pipe->InitBuffer(inQRingD, 1, STEP_W * STEP_V * sizeof(bfloat16_t));
        pipe->InitBuffer(inQU, 1, 44 * STEP_V * sizeof(bfloat16_t));
        // phi and fs share one queue slot: phiLen (32B-aligned) + W*fg
        pipe->InitBuffer(inQPhiFs, 1,
                         (STEP_P * STEP_K + 5 * 64 + STEP_W * 64)
                             * sizeof(bfloat16_t));
        pipe->InitBuffer(inQGate, 1, STEP_W * sizeof(float));
        pipe->InitBuffer(outBuf, STEP_V * sizeof(bfloat16_t));
        // 512B: largest staged payload is 128 floats (dc / currentD store)
        pipe->InitBuffer(outQStage, 1, STEP_V * sizeof(float));
        pipe->InitBuffer(calcBuf, TOTAL_CALC * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // diagnostic sub-modes: 100 = Init only, 101 = +qkv load,
        // 102 = +out store, 103 = full mode-0
        if (mode >= 100) {
            for (uint32_t i = 0; i < myCount; ++i) {
                uint32_t n = (myStart + i) / hvNum;
                uint32_t hv = (myStart + i) % hvNum;
                if (mode >= 101) {
                    LocalTensor<bfloat16_t> qUb = inQQkv.AllocTensor<bfloat16_t>();
                    DataCopy(qUb, gmQkv[(uint64_t)n * sQkv + iHof(hv)], STEP_K);
                    inQQkv.EnQue(qUb);
                    LocalTensor<bfloat16_t> qDiag = inQQkv.DeQue<bfloat16_t>();
                    inQQkv.FreeTensor(qDiag);
                }
                if (mode >= 102) {
                    LocalTensor<float> acc = calcBuf.Get<float>()[O_ACC];
                    Duplicate(acc, 0.0f, STEP_V);
                    PipeBarrier<PIPE_V>();
                    EmitOut(n, hv, acc);
                }
            }
            return;
        }
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }


    // Stage a 128-float payload through the out queue and store to gmOut.
    // The framework EnQue/DeQue pair supplies the V->MTE3 (and MTE3->V)
    // ordering that --cce-auto-sync does not create for TBuf scratch views.
    __aicore__ inline void EmitOut(uint64_t n, uint32_t hv,
                                   const LocalTensor<float>& payload)
    {
        LocalTensor<bfloat16_t> ob = outQStage.AllocTensor<bfloat16_t>();
        Cast(ob, payload, RoundMode::CAST_RINT, STEP_V);
        outQStage.EnQue(ob);
        LocalTensor<bfloat16_t> obDq = outQStage.DeQue<bfloat16_t>();
        DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], obDq,
                 STEP_V);
        outQStage.FreeTensor(obDq);
    }

    // V->S: scalar (S-pipe) GetValue reads of V-computed scalars are NOT
    // ordered by --cce-auto-sync (observed: early lanes stale, last lane
    // fresh); a manual V_S event is required at each scalar-read site.
    __aicore__ inline void SyncV2S()
    {
        SetFlag<HardEvent::V_S>(STEP_V_S_EVT);
        WaitFlag<HardEvent::V_S>(STEP_V_S_EVT);
    }

    __aicore__ inline uint32_t iHof(uint32_t hv)
    {
        return hv / (hvNum / hNum) * STEP_K;
    }

    // checkpoint emit: zero out + release the four input tensors
    __aicore__ inline void CheckpointEmit(
        uint64_t n, uint32_t hv, LocalTensor<bfloat16_t>& qIn,
        LocalTensor<bfloat16_t>& kIn, LocalTensor<bfloat16_t>& vIn,
        LocalTensor<float>& gateIn, LocalTensor<bfloat16_t>& ringKIn,
        LocalTensor<bfloat16_t>& ringDIn)
    {
        LocalTensor<float> calc = calcBuf.Get<float>();
        LocalTensor<float> acc = calc[O_ACC];
        LocalTensor<float> qFs = calc[O_QF];
        Duplicate(acc, 0.0f, STEP_V);
        PipeBarrier<PIPE_V>();
        DataCopy(acc[96], qFs, 8);
        PipeBarrier<PIPE_ALL>();
        EmitOut(n, hv, acc);
        inQQkv.FreeTensor(qIn);
        inQGate.FreeTensor(gateIn);
        inQRingK.FreeTensor(ringKIn);
        inQRingD.FreeTensor(ringDIn);
    }

private:
    __aicore__ inline void ProcessOne(uint64_t inst)
    {
        uint32_t n = inst / hvNum;
        uint32_t hv = inst % hvNum;
        uint32_t iH = hv / hvPerH;
        uint32_t row = n - rowsLo;

        // slots/cidx/wp were loaded from the tensor base: index by the
        // GLOBAL row n, not the local row offset
        int32_t slot = slotsUb.GetValue(n);
        int32_t wp = wpUb.GetValue(n * 2);
        int32_t cidx = cidxUb.GetValue(n);
        int32_t m = ranksUb.GetValue(hv);
        int32_t layU = layoutUb.GetValue(hv * 4);
        int32_t layPhi = layoutUb.GetValue(hv * 4 + 1);
        int32_t layFs = layoutUb.GetValue(hv * 4 + 2);
        int32_t fg = layoutUb.GetValue(hv * 4 + 3);
        float aVal = aUb.GetValue(row * hvNum + hv);
        float bVal = bUb.GetValue(row * hvNum + hv);

        LocalTensor<float> calc = calcBuf.Get<float>();
        LocalTensor<float> qF = calc[O_QF];
        LocalTensor<float> kF = calc[O_KF];
        LocalTensor<float> vF = calc[O_VF];
        LocalTensor<float> qn = calc[O_QN];
        LocalTensor<float> kn = calc[O_KN];
        LocalTensor<float> ringKF = calc[O_RINGK];
        LocalTensor<float> ringDF = calc[O_RINGD];
        LocalTensor<float> fsF = calc[O_FSF];
        LocalTensor<float> gateF = calc[O_GATE];
        LocalTensor<float> sQ = calc[O_SQ];
        LocalTensor<float> sK = calc[O_SK];
        LocalTensor<float> dc = calc[O_DC];
        LocalTensor<float> acc = calc[O_ACC];
        LocalTensor<float> tmp = calc[O_TMP];
        LocalTensor<float> tmp2 = calc[O_TMP2];
        LocalTensor<float> redDst = calc[O_RED];
        LocalTensor<float> expArgs = calc[O_EXP];
        LocalTensor<float> adF = calc[O_AD];
        LocalTensor<float> xq = calc[O_M1];
        LocalTensor<float> xk = calc[O_M1 + 64];
        LocalTensor<float> eq = calc[O_M1 + 2 * 64];
        LocalTensor<float> ek = calc[O_M1 + 3 * 64];
        LocalTensor<float> fcur = calc[O_M1 + 4 * 64];
        LocalTensor<float> xg = calc[O_M1 + 5 * 64];
        LocalTensor<float> gtqV = calc[O_M1 + 6 * 64];
        LocalTensor<float> gtkV = calc[O_M1 + 7 * 64];
        LocalTensor<float> obF = calc[O_HK2];
        LocalTensor<float> hkV = calc[O_HK];

        // ---- qkv load + gates: q/k/v share one queue slot (queue-count
        // reduction: the event pool mis-synchronizes excess queues) ----
        LocalTensor<bfloat16_t> qkvUb = inQQkv.AllocTensor<bfloat16_t>();
        LocalTensor<float> gateUb0 = inQGate.AllocTensor<float>();
        LocalTensor<bfloat16_t> ringKUb0 = inQRingK.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> ringDUb0 = inQRingD.AllocTensor<bfloat16_t>();
        DataCopy(qkvUb, gmQkv[(uint64_t)n * sQkv + iH * STEP_K], STEP_K);
        DataCopy(qkvUb[STEP_K],
                 gmQkv[(uint64_t)n * sQkv + hNum * STEP_K + iH * STEP_K],
                 STEP_K);
        DataCopy(qkvUb[2 * STEP_K],
                 gmQkv[(uint64_t)n * sQkv + 2 * hNum * STEP_K + hv * STEP_V],
                 STEP_V);
        DataCopy(gateUb0,
                 gmGCache[(uint64_t)slot * sGcSlot + hv * sGcHead], STEP_W);
        DataCopy(ringKUb0,
                 gmKCache[(uint64_t)slot * sKcSlot + iH * sKcHead],
                 STEP_W * STEP_K);
        DataCopy(ringDUb0,
                 gmDCache[(uint64_t)slot * sDcSlot + hv * sDcHead],
                 STEP_W * STEP_V);
        inQQkv.EnQue(qkvUb);
        inQGate.EnQue(gateUb0);
        inQRingK.EnQue(ringKUb0);
        inQRingD.EnQue(ringDUb0);
        LocalTensor<bfloat16_t> qIn = inQQkv.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> kIn = qIn[STEP_K];
        LocalTensor<bfloat16_t> vIn = qIn[2 * STEP_K];
        LocalTensor<float> gateIn = inQGate.DeQue<float>();
        LocalTensor<bfloat16_t> ringKIn = inQRingK.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> ringDIn = inQRingD.DeQue<bfloat16_t>();

        // phi / fs share one queue slot (phi at 0, fs at phiLen);
        // u keeps its own queue
        LocalTensor<bfloat16_t> phiUb;
        LocalTensor<bfloat16_t> fsUb;
        LocalTensor<bfloat16_t> uUb;
        const bool sketchHead = (m > 0);

        if (slot <= 0) {
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            EmitOut(n, hv, acc);
            inQQkv.FreeTensor(qIn);
            return;
        }

        if (mode == 0) {  // diag: meta reads + zero out only
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            EmitOut(n, hv, acc);
            inQQkv.FreeTensor(qIn);
            return;
        }

        // sketch state loads come after the padding early-exit so padding
        // rows leak no queue slots (e2e padding rows are common)
        if (sketchHead) {
            // m <= P heads keep only the merged map (m*K); phi gains section
            // is skipped entirely (xq/xk are overridden by tq/tk)
            uint32_t phiLen = (m <= STEP_P)
                ? m * STEP_K
                : (STEP_P * STEP_K + (STEP_P + 1) * fg);
            phiUb = inQPhiFs.AllocTensor<bfloat16_t>();
            uUb = inQU.AllocTensor<bfloat16_t>();
            DataCopyParams phiParams{1, static_cast<uint16_t>(phiLen * 2),
                                     0, 0};
            DataCopyPadParams phiPad{false, 0, 0, 0};
            DataCopyPad(phiUb,
                        gmPhi[(uint64_t)cidx * sSm + layPhi],
                        phiParams, phiPad);
            DataCopy(phiUb[phiLen], gmFs[(uint64_t)cidx * sSf + layFs],
                     STEP_W * fg);
            DataCopy(uUb, gmU[(uint64_t)cidx * sSu + layU], m * STEP_V);
            inQPhiFs.EnQue(phiUb);
            inQU.EnQue(uUb);
            phiUb = inQPhiFs.DeQue<bfloat16_t>();
            fsUb = phiUb[phiLen];
            uUb = inQU.DeQue<bfloat16_t>();
        }
        PipeBarrier<PIPE_ALL>();  // DIAG: hard sync after dequeues

        Cast(kF, kIn, RoundMode::CAST_NONE, STEP_K);
        Cast(qF, qIn, RoundMode::CAST_NONE, STEP_K);
        Cast(vF, vIn, RoundMode::CAST_NONE, STEP_V);
        Cast(obF, vF, RoundMode::CAST_RINT, STEP_V);
        Cast(ringKF, ringKIn, RoundMode::CAST_NONE, STEP_W * STEP_K);
        Cast(ringDF, ringDIn, RoundMode::CAST_NONE, STEP_W * STEP_V);
        DataCopy(gateF, gateIn, STEP_W);
        PipeBarrier<PIPE_V>();

        // zero stale ring rows >= wp (mirrors the triton masked loads)
        if (wp < STEP_W) {
            Duplicate(ringKF[wp * STEP_K], 0.0f, (STEP_W - wp) * STEP_K);
            Duplicate(ringDF[wp * STEP_V], 0.0f, (STEP_W - wp) * STEP_V);
            PipeBarrier<PIPE_V>();
        }
        if (mode == 20) {  // checkpoint: after loads + casts
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            EmitOut(n, hv, acc);
            inQQkv.FreeTensor(qIn);
            inQGate.FreeTensor(gateIn);
            inQRingK.FreeTensor(ringKIn);
            inQRingD.FreeTensor(ringDIn);
            return;
        }

        // ---- q/k norms and cur_kq ----
        Mul(tmp, qF, qF, STEP_K);
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(redDst, tmp, STEP_K);
        if (mode == 46) {  // diag: raw qF + raw q-sum, before scaling
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            Cast(tmp2, qIn, RoundMode::CAST_NONE, 16);
            PipeBarrier<PIPE_V>();
            acc.SetValue(0, redDst.GetValue(0));
            PipeBarrier<PIPE_V>();
            DataCopy(acc[8], qF, 8);
            DataCopy(acc[16], tmp2, 8);
            PipeBarrier<PIPE_ALL>();
            EmitOut(n, hv, acc);
            inQQkv.FreeTensor(qIn);
            inQGate.FreeTensor(gateIn);
            inQRingK.FreeTensor(ringKIn);
            inQRingD.FreeTensor(ringDIn);
            return;
        }
        PipeBarrier<PIPE_V>();
        SyncV2S();
        float qSc = scale / sqrt(redDst.GetValue(0) + 1e-6f);
        Mul(tmp, kF, kF, STEP_K);
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(redDst, tmp, STEP_K);
        PipeBarrier<PIPE_V>();
        SyncV2S();
        float kRn = 1.0f / sqrt(redDst.GetValue(0) + 1e-6f);
        Muls(qF, qF, qSc, STEP_K);
        PipeBarrier<PIPE_V>();
        Muls(kF, kF, kRn, STEP_K);
        PipeBarrier<PIPE_V>();
        Mul(tmp, qF, kF, STEP_K);
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(redDst, tmp, STEP_K);
        PipeBarrier<PIPE_V>();
        SyncV2S();
        float curKq = redDst.GetValue(0);

        // ---- gate prefix: pre[w] over rows < wp ----
        float pre[STEP_W];
        float accP = 0.0f;
        for (uint32_t w = 0; w < STEP_W; ++w) {
            accP += (w < wp) ? gateIn.GetValue(w) : 0.0f;
            pre[w] = accP;
        }
        float gtot = accP;

        // ---- scalar transcendentals via vector Exp/Log ----
        // expArgs pass A: [xg, -b, aLog, gtot] -> Exp
        float xgv = aVal + dtUb.GetValue(hv);
        expArgs.SetValue(0, xgv);
        expArgs.SetValue(1, -bVal);
        expArgs.SetValue(2, alogUb.GetValue(hv));
        expArgs.SetValue(3, gtot);
        PipeBarrier<PIPE_V>();
        AscendC::Exp(expArgs, expArgs, 8);
        PipeBarrier<PIPE_V>();
        SyncV2S();
        float exg = expArgs.GetValue(0);
        float eb = expArgs.GetValue(1);
        float eAl = expArgs.GetValue(2);
        float tot = expArgs.GetValue(3);
        // softplus via vector Log of (1 + exp(xg))
        float sp;
        expArgs.SetValue(0, 1.0f + exg);
        PipeBarrier<PIPE_V>();
        AscendC::Log(expArgs, expArgs, 8);
        PipeBarrier<PIPE_V>();
        SyncV2S();
        sp = (xgv <= 20.0f) ? expArgs.GetValue(0) : xgv;
        float gVal = -eAl * sp;
        // alpha = exp(gVal)
        expArgs.SetValue(0, gVal);
        PipeBarrier<PIPE_V>();
        AscendC::Exp(expArgs, expArgs, 8);
        PipeBarrier<PIPE_V>();
        SyncV2S();
        float alpha = expArgs.GetValue(0);
        float beta = 1.0f / (1.0f + eb);
        // match the triton reference: beta is rounded through bf16
        {
            LocalTensor<bfloat16_t> betaBf = outBuf.Get<bfloat16_t>();
            Duplicate(redDst, beta, 8);
            PipeBarrier<PIPE_V>();
            Cast(betaBf, redDst, RoundMode::CAST_RINT, 8);
            PipeBarrier<PIPE_V>();
            Cast(redDst, betaBf, RoundMode::CAST_NONE, 8);
            PipeBarrier<PIPE_V>();
            SyncV2S();
            beta = redDst.GetValue(0);
        }
        DataCopyParams cbParams;
        cbParams.blockCount = 1;
        cbParams.blockLen = 4;
        cbParams.srcStride = 0;
        cbParams.dstStride = 0;
        // beta/gVal live in registers; Duplicate them into the out queue so
        // the framework supplies the V->MTE3 ordering for the GM writes
        LocalTensor<float> scOut = outQStage.AllocTensor<float>();
        Duplicate(scOut, beta, 8);
        Duplicate(scOut[8], gVal, 8);
        outQStage.EnQue(scOut);
        LocalTensor<float> scOutDq = outQStage.DeQue<float>();
        DataCopyPad(gmBetaRing[(uint64_t)cidx * hvNum * STEP_W
                               + hv * STEP_W + wp],
                    scOutDq, cbParams);
        DataCopyPad(gmGCache[(uint64_t)slot * sGcSlot + hv * sGcHead + wp],
                    scOutDq[8], cbParams);
        outQStage.FreeTensor(scOutDq);

        if (mode == 21) {  // checkpoint: after gates + ring pad writes
            CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
            return;
        }

        // ---- k write: one v-head per key-head ----
        if (hv % hvPerH == 0) {
            LocalTensor<float> kSt = outQStage.AllocTensor<float>();
            if (wp == STEP_W - 1) {
                Muls(kSt, kF, 1.0f, STEP_K);
                outQStage.EnQue(kSt);
                LocalTensor<float> kDq = outQStage.DeQue<float>();
                DataCopy(gmCurrentK[(uint64_t)cidx * hNum * STEP_K
                                    + iH * STEP_K],
                         kDq, STEP_K);
                outQStage.FreeTensor(kDq);
            } else {
                LocalTensor<bfloat16_t> kBf = kSt.ReinterpretCast<bfloat16_t>();
                Cast(kBf, kF, RoundMode::CAST_RINT, STEP_K);
                outQStage.EnQue(kSt);
                LocalTensor<float> kDq = outQStage.DeQue<float>();
                DataCopy(gmKCache[(uint64_t)slot * sKcSlot + iH * sKcHead
                                  + wp * STEP_K],
                         kDq.ReinterpretCast<bfloat16_t>(), STEP_K);
                outQStage.FreeTensor(kDq);
            }
        }
        if (mode == 22) {  // checkpoint: after k write
            CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
            return;
        }

        if (mode == 1) {  // diag: emit v
            EmitOut(n, hv, obF);
            inQQkv.FreeTensor(qIn);
            inQGate.FreeTensor(gateIn);
            inQRingK.FreeTensor(ringKIn);
            inQRingD.FreeTensor(ringDIn);
            return;
        }

        // ---- ring projections: kq_s / kk_s per w ----
        // per-w dot results are staged to vector windows (O_QN/O_KN, both
        // unused) via V-side copies; a pipe barrier precedes the reads
        float kqs[STEP_W];
        float kks[STEP_W];
        for (uint32_t w = 0; w < STEP_W; ++w) {
            Mul(tmp, ringKF[w * STEP_K], qF, STEP_K);
            PipeBarrier<PIPE_V>();
            ReduceSumHalfInterval(redDst, tmp, STEP_K);
            Muls(calc[O_QN + w * 8], redDst, 1.0f, 8);
            Mul(tmp, ringKF[w * STEP_K], kF, STEP_K);
            PipeBarrier<PIPE_V>();
            ReduceSumHalfInterval(redDst, tmp, STEP_K);
            Muls(calc[O_KN + w * 8], redDst, 1.0f, 8);
        }
        PipeBarrier<PIPE_V>();
        PipeBarrier<PIPE_V>();
        SyncV2S();
        for (uint32_t w = 0; w < STEP_W; ++w) {
            kqs[w] = calc[O_QN + w * 8].GetValue(0);
            kks[w] = calc[O_KN + w * 8].GetValue(0);
        }

        // ---- rep[w] = exp(gtot - pre[w]) for w < wp, else ~0 ----
        for (uint32_t w = 0; w < STEP_W; ++w) {
            expArgs.SetValue(w, (w < wp) ? (gtot - pre[w]) : -30.0f);
        }
        PipeBarrier<PIPE_V>();
        AscendC::Exp(expArgs, expArgs, 32);
        PipeBarrier<PIPE_V>();

        // ---- s_q / s_k: sum_w d[w] * (kq/kk)[w] * rep[w] ----
        Duplicate(sQ, 0.0f, STEP_V);
        Duplicate(sK, 0.0f, STEP_V);
        PipeBarrier<PIPE_V>();
        SyncV2S();
        for (uint32_t w = 0; w < STEP_W; ++w) {
            float wq = kqs[w] * expArgs.GetValue(w);
            float wk = kks[w] * expArgs.GetValue(w);
            // expArgs was synced above; kqs/kks synced at their reads
            if (wq != 0.0f) {
                Muls(tmp, ringDF[w * STEP_V], wq, STEP_V);
                PipeBarrier<PIPE_V>();
                Add(sQ, sQ, tmp, STEP_V);
                PipeBarrier<PIPE_V>();
            }
            if (wk != 0.0f) {
                Muls(tmp, ringDF[w * STEP_V], wk, STEP_V);
                PipeBarrier<PIPE_V>();
                Add(sK, sK, tmp, STEP_V);
                PipeBarrier<PIPE_V>();
            }
        }
        if (mode == 23) {  // checkpoint: after s_q/s_k
            CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
            return;
        }

        if (m > 0) {
            // ---- phi / fs / u: dequeued up top with the other inputs ----
            Cast(fsF, fsUb, RoundMode::CAST_NONE, STEP_W * fg);
            PipeBarrier<PIPE_V>();
            if (mode == 47) {  // diag: raw loaded fs payload (first 128)
                DataCopy(acc, fsF, STEP_V);
                PipeBarrier<PIPE_V>();
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                EmitOut(n, hv, acc);
                inQQkv.FreeTensor(qIn);
                inQGate.FreeTensor(gateIn);
                inQRingK.FreeTensor(ringKIn);
                inQRingD.FreeTensor(ringDIn);
                return;
            }
            if (mode == 48) {  // diag: merged phi row 0 as float
                Cast(tmp, phiUb, RoundMode::CAST_NONE, STEP_K);
                PipeBarrier<PIPE_V>();
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                EmitOut(n, hv, tmp);
                inQQkv.FreeTensor(qIn);
                inQGate.FreeTensor(gateIn);
                inQRingK.FreeTensor(ringKIn);
                inQRingD.FreeTensor(ringDIn);
                return;
            }
            if (mode == 26) {  // checkpoint: after phi/fs/u loads
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // pivots: tq[p] / tk[p] = <phi_p, q/k>
            LocalTensor<float> pivF = tmp2;
            float tq[STEP_P];
            float tk[STEP_P];
            for (uint32_t p = 0; p < STEP_P; ++p) {
                Cast(pivF, phiUb[p * STEP_K], RoundMode::CAST_NONE, STEP_K);
                PipeBarrier<PIPE_V>();
                Mul(tmp, pivF, qF, STEP_K);
                PipeBarrier<PIPE_V>();
                ReduceSumHalfInterval(redDst, tmp, STEP_K);
                Muls(gtqV[p * 8], redDst, 1.0f, 8);
                Mul(tmp, pivF, kF, STEP_K);
                PipeBarrier<PIPE_V>();
                ReduceSumHalfInterval(redDst, tmp, STEP_K);
                Muls(gtkV[p * 8], redDst, 1.0f, 8);
            }
            PipeBarrier<PIPE_V>();
            PipeBarrier<PIPE_V>();
            SyncV2S();
            for (uint32_t p = 0; p < STEP_P; ++p) {
                tq[p] = gtqV[p * 8].GetValue(0);
                tk[p] = gtkV[p * 8].GetValue(0);
            }
            if (mode == 27) {  // checkpoint: after pivots
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // gains projection into xq / xk (m-wide)
            Cast(adF, phiUb[STEP_P * STEP_K], RoundMode::CAST_NONE, m);
            PipeBarrier<PIPE_V>();
            if (mode == 32) {
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }
            Duplicate(xq, 0.0f, m);
            Duplicate(xk, 0.0f, m);
            PipeBarrier<PIPE_V>();
            if (mode == 33) {
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }
            for (uint32_t p = 0; p < STEP_P; ++p) {
                Cast(tmp2, phiUb[STEP_P * STEP_K + (p + 1) * fg],
                     RoundMode::CAST_NONE, m);
                PipeBarrier<PIPE_V>();
                if (mode == 35 && p == 0) {
                    inQPhiFs.FreeTensor(phiUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
                Muls(tmp, tmp2, tq[p], m);
                PipeBarrier<PIPE_V>();
                if (mode == 36 && p == 0) {
                    inQPhiFs.FreeTensor(phiUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
                Add(xq, xq, tmp, m);
                PipeBarrier<PIPE_V>();
                if (mode == 37 && p == 0) {
                    inQPhiFs.FreeTensor(phiUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
                Muls(tmp, tmp2, tk[p], m);
                PipeBarrier<PIPE_V>();
                Add(xk, xk, tmp, m);
                PipeBarrier<PIPE_V>();
                if (mode == 34 && p == 0) {
                    inQPhiFs.FreeTensor(phiUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
            }
            if (mode == 29) {  // checkpoint: after gains p-loop
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }
            // + ad * qn[0:m] / kn[0:m]  (qF/kF hold the normalized q/k)
            Mul(tmp, qF, adF, m);
            PipeBarrier<PIPE_V>();
            Add(xq, xq, tmp, m);
            PipeBarrier<PIPE_V>();
            Mul(tmp, kF, adF, m);
            PipeBarrier<PIPE_V>();
            Add(xk, xk, tmp, m);
            PipeBarrier<PIPE_V>();
            // merged-map override for m <= P: xq_g = tq_g, xk_g = tk_g
            if (m <= STEP_P) {
                for (uint32_t g = 0; g < (uint32_t)m; ++g) {
                    xq.SetValue(g, tq[g]);
                    xk.SetValue(g, tk[g]);
                }
                PipeBarrier<PIPE_V>();
            }
            if (mode == 30) {  // checkpoint: after qn/kn adds + override
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // eq/ek: sum_w fs[w] * kq/kk[w] (rows w < wp)
            Duplicate(eq, 0.0f, m);
            Duplicate(ek, 0.0f, m);
            PipeBarrier<PIPE_V>();
            for (uint32_t w = 0; w < (uint32_t)wp; ++w) {
                Muls(tmp, fsF[w * fg], kqs[w], m);
                PipeBarrier<PIPE_V>();
                Add(eq, eq, tmp, m);
                PipeBarrier<PIPE_V>();
                Muls(tmp, fsF[w * fg], kks[w], m);
                PipeBarrier<PIPE_V>();
                Add(ek, ek, tmp, m);
                PipeBarrier<PIPE_V>();
            }
            if (mode == 31) {  // checkpoint: after eq/ek
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }
            if (mode == 49) {  // diag: ek[0:64] + xk[0:64] pre-fcur
                DataCopy(acc, ek, 64);
                DataCopy(acc[64], xk, 64);
                PipeBarrier<PIPE_V>();
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                EmitOut(n, hv, acc);
                inQQkv.FreeTensor(qIn);
                inQGate.FreeTensor(gateIn);
                inQRingK.FreeTensor(ringKIn);
                inQRingD.FreeTensor(ringDIn);
                return;
            }

            // fcur = beta*(xk - ek); x = xq - eq - fcur*curKq
            Sub(xk, xk, ek, m);
            PipeBarrier<PIPE_V>();
            Muls(fcur, xk, beta, m);
            PipeBarrier<PIPE_V>();
            Muls(tmp, fcur, curKq, m);
            PipeBarrier<PIPE_V>();
            Sub(xg, xq, eq, m);
            PipeBarrier<PIPE_V>();
            Sub(xg, xg, tmp, m);
            PipeBarrier<PIPE_V>();
            if (mode == 28) {  // checkpoint: after eq/ek/fcur/x
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // fs[wp] = fcur (bf16, m elems).  Stage through the out queue:
            // the framework inserts a real V->MTE3 event that auto-sync fails
            // to create for reinterpret-cast scratch views.
            LocalTensor<bfloat16_t> fcurBf = outQStage.AllocTensor<bfloat16_t>();
            Cast(fcurBf, fcur, RoundMode::CAST_RINT, 64);
            outQStage.EnQue(fcurBf);
            LocalTensor<bfloat16_t> fcurOut = outQStage.DeQue<bfloat16_t>();
            DataCopyParams fsParams{1, static_cast<uint16_t>(m * 2), 0, 0};
            DataCopyPad(gmFs[(uint64_t)cidx * sSf + layFs + wp * fg],
                        fcurOut, fsParams);
            outQStage.FreeTensor(fcurOut);
            if (mode == 24) {  // checkpoint: after phi/fs/u math
                inQPhiFs.FreeTensor(phiUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // hq = sum_g u[g] * x[g]  (V-wide)
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            SyncV2S();
            for (uint32_t g = 0; g < (uint32_t)m; ++g) {
                float x = xg.GetValue(g);
                Cast(tmp, uUb[g * STEP_V], RoundMode::CAST_NONE, STEP_V);
                PipeBarrier<PIPE_V>();
                Muls(tmp, tmp, x, STEP_V);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, tmp, STEP_V);
                PipeBarrier<PIPE_V>();
            }
            inQPhiFs.FreeTensor(phiUb);
            inQU.FreeTensor(uUb);

            // dc = beta * (v - alpha * s_k)
            Muls(tmp, sK, alpha, STEP_V);
            PipeBarrier<PIPE_V>();
            Sub(tmp, vF, tmp, STEP_V);
            PipeBarrier<PIPE_V>();
            Muls(dc, tmp, beta, STEP_V);
            PipeBarrier<PIPE_V>();
            // out = alpha * (hq * tot + s_q) + dc * cur_kq
            Muls(tmp, acc, tot, STEP_V);
            PipeBarrier<PIPE_V>();
            Add(tmp, tmp, sQ, STEP_V);
            PipeBarrier<PIPE_V>();
            Muls(tmp, tmp, alpha, STEP_V);
            PipeBarrier<PIPE_V>();
            Muls(tmp2, dc, curKq, STEP_V);
            PipeBarrier<PIPE_V>();
            Add(tmp, tmp, tmp2, STEP_V);
            PipeBarrier<PIPE_V>();
        } else {
            // ---- dense head: hq/hk from the full state (V rows of K) ----
        LocalTensor<float> hkV = calc[O_HK];
            Duplicate(acc, 0.0f, STEP_V);
            Duplicate(hkV, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            for (uint32_t v = 0; v < STEP_V; ++v) {
                DataCopy(tmp,
                         gmState[(uint64_t)slot * sStSlot + hv * sStHead
                                 + v * STEP_K],
                         STEP_K);
                PipeBarrier<PIPE_ALL>();
                Mul(tmp2, tmp, qF, STEP_K);
                PipeBarrier<PIPE_V>();
                ReduceSumHalfInterval(redDst, tmp2, STEP_K);
                acc.SetValue(v, redDst.GetValue(0));
                Mul(tmp2, tmp, kF, STEP_K);
                PipeBarrier<PIPE_V>();
                ReduceSumHalfInterval(redDst, tmp2, STEP_K);
                hkV.SetValue(v, redDst.GetValue(0));
            }
            // dc = beta * (v - alpha * (hk * tot + s_k))
            Muls(tmp, hkV, alpha * tot, STEP_V);
            PipeBarrier<PIPE_V>();
            Add(tmp, tmp, sK, STEP_V);
            PipeBarrier<PIPE_V>();
            Muls(tmp, tmp, alpha, STEP_V);
            PipeBarrier<PIPE_V>();
            Sub(tmp, vF, tmp, STEP_V);
            PipeBarrier<PIPE_V>();
            Muls(dc, tmp, beta, STEP_V);
            PipeBarrier<PIPE_V>();
            // out = alpha * (hq * tot + s_q) + dc * cur_kq
            Muls(tmp, acc, tot, STEP_V);
            PipeBarrier<PIPE_V>();
            Add(tmp, tmp, sQ, STEP_V);
            PipeBarrier<PIPE_V>();
            Muls(tmp, tmp, alpha, STEP_V);
            PipeBarrier<PIPE_V>();
            Muls(tmp2, dc, curKq, STEP_V);
            PipeBarrier<PIPE_V>();
            Add(tmp, tmp, tmp2, STEP_V);
            PipeBarrier<PIPE_V>();
        }

        if (mode == 25) {  // checkpoint: after dc/out math, before stores
            CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
            return;
        }

        if (mode == 45) {  // diag: pack dc/s_q/s_k/scalars into out
            // out[0:32]=dc, [32:64]=s_q, [64:96]=s_k, [96]=cur_kq,
            // [97]=alpha, [98]=beta, [99]=tot
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            acc.SetValue(0, curKq);
            acc.SetValue(1, alpha);
            acc.SetValue(2, beta);
            acc.SetValue(3, gtot);
            PipeBarrier<PIPE_V>();
            DataCopy(tmp, dc, STEP_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(tmp[32], sQ, 32);
            DataCopy(tmp[64], sK, 32);
            PipeBarrier<PIPE_ALL>();
            tmp.SetValue(96, curKq);
            tmp.SetValue(97, alpha);
            tmp.SetValue(98, beta);
            tmp.SetValue(99, gtot);
            tmp.SetValue(100, qSc);
            tmp.SetValue(101, kRn);
            tmp.SetValue(102, kqs[0]);
            tmp.SetValue(103, kks[0]);
            for (uint32_t z = 0; z < 4; ++z) {
                tmp.SetValue(104 + z, qF.GetValue(z));
                tmp.SetValue(108 + z, kF.GetValue(z));
            }
            PipeBarrier<PIPE_V>();
            EmitOut(n, hv, tmp);
            inQQkv.FreeTensor(qIn);
            inQGate.FreeTensor(gateIn);
            inQRingK.FreeTensor(ringKIn);
            inQRingD.FreeTensor(ringDIn);
            return;
        }

        // ---- d store + out store ----
        if (wp == STEP_W - 1) {
            LocalTensor<float> dSt = outQStage.AllocTensor<float>();
            Muls(dSt, dc, 1.0f, STEP_V);
            outQStage.EnQue(dSt);
            LocalTensor<float> dDq = outQStage.DeQue<float>();
            DataCopy(gmCurrentD[(uint64_t)cidx * hvNum * STEP_V
                                + hv * STEP_V],
                     dDq, STEP_V);
            outQStage.FreeTensor(dDq);
        } else {
            LocalTensor<float> dSt = outQStage.AllocTensor<float>();
            LocalTensor<bfloat16_t> dcBf = dSt.ReinterpretCast<bfloat16_t>();
            Cast(dcBf, dc, RoundMode::CAST_RINT, STEP_V);
            outQStage.EnQue(dSt);
            LocalTensor<float> dDq = outQStage.DeQue<float>();
            DataCopy(gmDCache[(uint64_t)slot * sDcSlot + hv * sDcHead
                              + wp * STEP_V],
                     dDq.ReinterpretCast<bfloat16_t>(), STEP_V);
            outQStage.FreeTensor(dDq);
        }
        if (mode == 44) {  // diag: full flow, pack diagnostics into out
            // [0:8]=s_q[0:8] [8:16]=s_k[0:8] [16:24]=dc[0:8]
            // [24]=qSc [25]=kRn [26]=curKq [27]=kqs[0] [28]=kks[0]
            // [29]=alpha [30]=beta [31]=gtot
            Duplicate(tmp2, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            DataCopy(tmp2[0], sQ, 8);
            DataCopy(tmp2[8], sK, 8);
            DataCopy(tmp2[16], dc, 8);
            PipeBarrier<PIPE_ALL>();
            tmp2.SetValue(23, this->scale);
            tmp2.SetValue(24, qSc);
            tmp2.SetValue(25, kRn);
            tmp2.SetValue(26, curKq);
            tmp2.SetValue(27, kqs[0]);
            tmp2.SetValue(28, kks[0]);
            tmp2.SetValue(29, alpha);
            tmp2.SetValue(30, beta);
            tmp2.SetValue(31, gtot);
            DataCopy(tmp2[40], dc[96], 32);
            DataCopy(tmp2[72], sQ[96], 32);
            PipeBarrier<PIPE_V>();
            EmitOut(n, hv, tmp2);
        } else {
            EmitOut(n, hv, tmp);
        }

        inQQkv.FreeTensor(qIn);
        inQGate.FreeTensor(gateIn);
        inQRingK.FreeTensor(ringKIn);
        inQRingD.FreeTensor(ringDIn);
    }

    GlobalTensor<bfloat16_t> gmQkv, gmDCache, gmKCache, gmU, gmPhi, gmFs,
        gmOut;
    GlobalTensor<float> gmA, gmB, gmALog, gmDtBias, gmState, gmGCache;
    GlobalTensor<int32_t> gmSlots, gmMeta, gmRanks, gmLayout;
    GlobalTensor<int32_t> gmWritePos;
    GlobalTensor<float> gmBetaRing, gmCurrentD, gmCurrentK;
    TQue<QuePosition::VECIN, 1> inQRingK, inQRingD, inQPhiFs, inQGate, inQQkv;
    TQue<QuePosition::VECIN, 1> inQU;
    TQue<QuePosition::VECOUT, 1> outQStage;
    TBuf<QuePosition::VECCALC> calcBuf, outBuf;
    TBuf<QuePosition::VECCALC> metaSlots, metaWp, metaCidx, metaA, metaB;
    TBuf<QuePosition::VECCALC> metaALog, metaDt, metaRanks, metaLayout;
    LocalTensor<int32_t> slotsUb, cidxUb, ranksUb, layoutUb;
    LocalTensor<int32_t> wpUb;
    LocalTensor<float> aUb, bUb, alogUb, dtUb;
    static constexpr uint32_t STEP_V_S_EVT = 2;
    uint32_t myStart = 0, myCount = 0, hvNum = 0, hNum = 0, hvPerH = 2;
    uint32_t mode = 99;
    uint32_t rowsLo = 0, nRows = 0;
    float scale = 0.f;
    int64_t sQkv = 0, sA = 0, sB = 0;
    int64_t sStSlot = 0, sStHead = 0, sDcSlot = 0, sDcHead = 0;
    int64_t sKcSlot = 0, sKcHead = 0, sGcSlot = 0, sGcHead = 0;
    int64_t sSu = 0, sSm = 0, sSf = 0;

    // calc buffer layout (floats)
    static constexpr uint32_t O_QF = 0;
    static constexpr uint32_t O_KF = O_QF + STEP_K;
    static constexpr uint32_t O_VF = O_KF + STEP_K;
    static constexpr uint32_t O_QN = O_VF + STEP_V;
    static constexpr uint32_t O_KN = O_QN + STEP_K;
    static constexpr uint32_t O_RINGK = O_KN + STEP_K;
    static constexpr uint32_t O_RINGD = O_RINGK + STEP_W * STEP_K;
    static constexpr uint32_t O_FSF = O_RINGD + STEP_W * STEP_V;
    static constexpr uint32_t O_GATE = O_FSF + STEP_W * 64;
    static constexpr uint32_t O_SQ = O_GATE + STEP_W;
    static constexpr uint32_t O_SK = O_SQ + STEP_V;
    static constexpr uint32_t O_DC = O_SK + STEP_V;
    static constexpr uint32_t O_ACC = O_DC + STEP_V;
    static constexpr uint32_t O_TMP = O_ACC + STEP_V;
    static constexpr uint32_t O_TMP2 = O_TMP + STEP_V;
    static constexpr uint32_t O_RED = O_TMP2 + STEP_V;
    static constexpr uint32_t O_EXP = O_RED + 64;
    static constexpr uint32_t O_AD = O_EXP + 64;
    static constexpr uint32_t O_M1 = O_AD + 64;
    static constexpr uint32_t O_HK = O_M1 + 8 * 64;
    static constexpr uint32_t O_HK2 = O_HK + STEP_V;
    static constexpr uint32_t TOTAL_CALC = O_HK2 + STEP_V;
};

extern "C" __global__ __aicore__ void gdn_sketch_step(
    GM_ADDR mixedQkv, GM_ADDR aAct, GM_ADDR bAct, GM_ADDR aLog,
    GM_ADDR dtBias, GM_ADDR state, GM_ADDR dCache, GM_ADDR kCache,
    GM_ADDR gCache, GM_ADDR slots, GM_ADDR writePos, GM_ADDR meta, GM_ADDR u,
    GM_ADDR phi, GM_ADDR fs, GM_ADDR betaRing, GM_ADDR currentD,
    GM_ADDR currentK, GM_ADDR ranks, GM_ADDR layout, GM_ADDR out,
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
    GdnStepKernel op;
    op.Init(mixedQkv, aAct, bAct, aLog, dtBias, state, dCache, kCache,
            gCache, slots, writePos, meta, u, phi, fs, betaRing, currentD,
            currentK, ranks, layout, out, myStart, myCount,
            tilingData.hv, tilingData.h, tilingData.scale, tilingData.mode,
            tilingData.sQkv,
            tilingData.sA, tilingData.sB, tilingData.sStSlot,
            tilingData.sStHead, tilingData.sDcSlot, tilingData.sDcHead,
            tilingData.sKcSlot, tilingData.sKcHead, tilingData.sGcSlot,
            tilingData.sGcHead, tilingData.sSu, tilingData.sSm,
            tilingData.sSf, &pipe);
    op.Process();
}
