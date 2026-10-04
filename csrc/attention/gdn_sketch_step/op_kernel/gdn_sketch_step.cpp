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
        // load from the aligned tensor base with a 32B-multiple length
        DataCopy(slotsUb, gmSlots[0], rowsPad);
        DataCopy(cidxUb, gmMeta[0], rowsPad);
        // write_pos is int64; read the low 32-bit word of each row
        DataCopy(wpUb, gmWritePos[0], rowsPad * 2);
        DataCopy(aUb, gmA[(uint64_t)rowsLo * hvNum], nRows * hvNum);
        DataCopy(bUb, gmB[(uint64_t)rowsLo * hvNum], nRows * hvNum);
        DataCopy(alogUb, gmALog[0], hvNum);
        DataCopy(dtUb, gmDtBias[0], hvNum);
        DataCopy(ranksUb, gmRanks[0], hvNum);
        DataCopy(layoutUb, gmLayout[0], hvNum * 4);
        PipeBarrier<PIPE_ALL>();

        pipe->InitBuffer(inQRingK, 1, STEP_W * STEP_K * sizeof(bfloat16_t));
        pipe->InitBuffer(inQRingD, 1, STEP_W * STEP_V * sizeof(bfloat16_t));
        pipe->InitBuffer(inQU, 1, 64 * STEP_V * sizeof(bfloat16_t));
        pipe->InitBuffer(inQPhi, 1,
                         (STEP_P * STEP_K + 5 * 64) * sizeof(bfloat16_t));
        pipe->InitBuffer(inQFs, 1, STEP_W * 64 * sizeof(bfloat16_t));
        pipe->InitBuffer(inQGate, 1, STEP_W * sizeof(float));
        pipe->InitBuffer(inQQ, 1, STEP_K * sizeof(bfloat16_t));
        pipe->InitBuffer(inQK, 1, STEP_K * sizeof(bfloat16_t));
        pipe->InitBuffer(inQV, 1, STEP_V * sizeof(bfloat16_t));
        pipe->InitBuffer(outBuf, STEP_V * sizeof(bfloat16_t));
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
                    LocalTensor<bfloat16_t> qUb = inQQ.AllocTensor<bfloat16_t>();
                    DataCopy(qUb, gmQkv[(uint64_t)n * sQkv + iHof(hv)], STEP_K);
                    inQQ.EnQue(qUb);
                    LocalTensor<bfloat16_t> qDiag = inQQ.DeQue<bfloat16_t>();
                    inQQ.FreeTensor(qDiag);
                }
                if (mode >= 102) {
                    LocalTensor<float> acc = calcBuf.Get<float>()[O_ACC];
                    Duplicate(acc, 0.0f, STEP_V);
                    PipeBarrier<PIPE_V>();
                    LocalTensor<bfloat16_t> ob = outBuf.Get<bfloat16_t>();
                    Cast(ob, acc, RoundMode::CAST_RINT, STEP_V);
                    PipeBarrier<PIPE_V>();
                    DataCopy(gmOut[(uint64_t)(n * hvNum + hv) * STEP_V], ob,
                             STEP_V);
                }
            }
            return;
        }
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }

    // V->MTE3: make all prior V results visible to the MTE3 engine
    __aicore__ inline void SyncV2Mte3()
    {
        // DIAG: disabled
    }

    // MTE3->V: MTE3 has finished reading scratch that V is about to reuse
    __aicore__ inline void SyncMte32V()
    {
        // DIAG: disabled
    }

    // V->S: make V-pipe results visible to scalar (S-pipe) UB reads
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
        LocalTensor<bfloat16_t> ob = outBuf.Get<bfloat16_t>();
        Duplicate(acc, 0.0f, STEP_V);
        PipeBarrier<PIPE_V>();
        DataCopy(acc[96], qFs, 8);
        PipeBarrier<PIPE_ALL>();
        Cast(ob, acc, RoundMode::CAST_RINT, STEP_V);
        PipeBarrier<PIPE_V>();
        DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], ob, STEP_V);
        inQQ.FreeTensor(qIn);
        inQK.FreeTensor(kIn);
        inQV.FreeTensor(vIn);
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

        // ---- qkv load + gates (M0-probe pattern: alloc all, copy all,
        // enque all, deque all) ----
        LocalTensor<bfloat16_t> qUb = inQQ.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> kUb = inQK.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> vUb = inQV.AllocTensor<bfloat16_t>();
        LocalTensor<float> gateUb0 = inQGate.AllocTensor<float>();
        LocalTensor<bfloat16_t> ringKUb0 = inQRingK.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> ringDUb0 = inQRingD.AllocTensor<bfloat16_t>();
        DataCopy(qUb, gmQkv[(uint64_t)n * sQkv + iH * STEP_K], STEP_K);
        DataCopy(kUb,
                 gmQkv[(uint64_t)n * sQkv + hNum * STEP_K + iH * STEP_K],
                 STEP_K);
        DataCopy(vUb,
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
        inQQ.EnQue(qUb);
        inQK.EnQue(kUb);
        inQV.EnQue(vUb);
        inQGate.EnQue(gateUb0);
        inQRingK.EnQue(ringKUb0);
        inQRingD.EnQue(ringDUb0);
        LocalTensor<bfloat16_t> qIn = inQQ.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> kIn = inQK.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> vIn = inQV.DeQue<bfloat16_t>();
        LocalTensor<float> gateIn = inQGate.DeQue<float>();
        LocalTensor<bfloat16_t> ringKIn = inQRingK.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> ringDIn = inQRingD.DeQue<bfloat16_t>();
        PipeBarrier<PIPE_ALL>();  // DIAG: hard sync after dequeues

        if (slot <= 0) {
            LocalTensor<bfloat16_t> ob = outBuf.Get<bfloat16_t>();
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            Cast(ob, acc, RoundMode::CAST_RINT, STEP_V);
            PipeBarrier<PIPE_V>();
            DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], ob,
                     STEP_V);
            inQQ.FreeTensor(qIn);
            inQK.FreeTensor(kIn);
            inQV.FreeTensor(vIn);
            return;
        }

        if (mode == 0) {  // diag: meta reads + zero out only
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            LocalTensor<bfloat16_t> ob0 = outBuf.Get<bfloat16_t>();
            Cast(ob0, acc, RoundMode::CAST_RINT, STEP_V);
            PipeBarrier<PIPE_V>();
            DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], ob0,
                     STEP_V);
            inQQ.FreeTensor(qIn);
            inQK.FreeTensor(kIn);
            inQV.FreeTensor(vIn);
            return;
        }

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
            LocalTensor<bfloat16_t> odb = outBuf.Get<bfloat16_t>();
            Duplicate(acc, 0.0f, STEP_V);
            PipeBarrier<PIPE_V>();
            Cast(odb, acc, RoundMode::CAST_RINT, STEP_V);
            PipeBarrier<PIPE_V>();
            DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], odb,
                     STEP_V);
            inQQ.FreeTensor(qIn);
            inQK.FreeTensor(kIn);
            inQV.FreeTensor(vIn);
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
            LocalTensor<bfloat16_t> ob = outBuf.Get<bfloat16_t>();
            Cast(ob, acc, RoundMode::CAST_RINT, STEP_V);
            PipeBarrier<PIPE_V>();
            DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], ob,
                     STEP_V);
            inQQ.FreeTensor(qIn);
            inQK.FreeTensor(kIn);
            inQV.FreeTensor(vIn);
            inQGate.FreeTensor(gateIn);
            inQRingK.FreeTensor(ringKIn);
            inQRingD.FreeTensor(ringDIn);
            return;
        }
        SyncV2S();
        float qSc = scale / sqrt(redDst.GetValue(0) + 1e-6f);
        Mul(tmp, kF, kF, STEP_K);
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(redDst, tmp, STEP_K);
        SyncV2S();
        float kRn = 1.0f / sqrt(redDst.GetValue(0) + 1e-6f);
        Muls(qF, qF, qSc, STEP_K);
        PipeBarrier<PIPE_V>();
        Muls(kF, kF, kRn, STEP_K);
        PipeBarrier<PIPE_V>();
        Mul(tmp, qF, kF, STEP_K);
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(redDst, tmp, STEP_K);
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
        SyncV2S();
        sp = (xgv <= 20.0f) ? expArgs.GetValue(0) : xgv;
        float gVal = -eAl * sp;
        // alpha = exp(gVal)
        expArgs.SetValue(0, gVal);
        PipeBarrier<PIPE_V>();
        AscendC::Exp(expArgs, expArgs, 8);
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
            SyncV2S();
            beta = redDst.GetValue(0);
        }
        DataCopyParams cbParams;
        cbParams.blockCount = 1;
        cbParams.blockLen = 4;
        cbParams.srcStride = 0;
        cbParams.dstStride = 0;
        // lanes 24/32: untouched by Exp(8) and by the rep loop (0..15);
        // written once per instance so MTE3 reads a stable source
        expArgs.SetValue(24, beta);
        expArgs.SetValue(32, gVal);
        PipeBarrier<PIPE_V>();
        SyncV2Mte3();
        DataCopyPad(gmBetaRing[(uint64_t)cidx * hvNum * STEP_W
                               + hv * STEP_W + wp],
                    expArgs[24], cbParams);
        DataCopyPad(gmGCache[(uint64_t)slot * sGcSlot + hv * sGcHead + wp],
                    expArgs[32], cbParams);

        if (mode == 21) {  // checkpoint: after gates + ring pad writes
            CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
            return;
        }

        // ---- k write: one v-head per key-head ----
        if (hv % hvPerH == 0) {
            if (wp == STEP_W - 1) {
                SyncV2Mte3();
                DataCopy(gmCurrentK[(uint64_t)cidx * hNum * STEP_K
                                    + iH * STEP_K],
                         kF, STEP_K);
                SyncMte32V();
            } else {
                LocalTensor<bfloat16_t> kBf =
                    calc[O_HK2].ReinterpretCast<bfloat16_t>();
                Cast(kBf, kF, RoundMode::CAST_RINT, STEP_K);
                SyncV2Mte3();
                DataCopy(gmKCache[(uint64_t)slot * sKcSlot + iH * sKcHead
                                  + wp * STEP_K],
                         kBf, STEP_K);
                SyncMte32V();
            }
        }
        if (mode == 22) {  // checkpoint: after k write
            CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
            return;
        }

        if (mode == 1) {  // diag: emit v
            LocalTensor<bfloat16_t> odb = outBuf.Get<bfloat16_t>();
            Cast(odb, obF, RoundMode::CAST_RINT, STEP_V);
            PipeBarrier<PIPE_V>();
            DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V],
                     odb, STEP_V);
            inQQ.FreeTensor(qIn);
            inQK.FreeTensor(kIn);
            inQV.FreeTensor(vIn);
            inQGate.FreeTensor(gateIn);
            inQRingK.FreeTensor(ringKIn);
            inQRingD.FreeTensor(ringDIn);
            return;
        }

        // ---- ring projections: kq_s / kk_s per w ----
        float kqs[STEP_W];
        float kks[STEP_W];
        for (uint32_t w = 0; w < STEP_W; ++w) {
            Mul(tmp, ringKF[w * STEP_K], qF, STEP_K);
            PipeBarrier<PIPE_V>();
            ReduceSumHalfInterval(redDst, tmp, STEP_K);
            SyncV2S();
            kqs[w] = redDst.GetValue(0);
            Mul(tmp, ringKF[w * STEP_K], kF, STEP_K);
            PipeBarrier<PIPE_V>();
            ReduceSumHalfInterval(redDst, tmp, STEP_K);
            SyncV2S();
            kks[w] = redDst.GetValue(0);
        }

        // ---- rep[w] = exp(gtot - pre[w]) for w < wp, else ~0 ----
        for (uint32_t w = 0; w < STEP_W; ++w) {
            expArgs.SetValue(w, (w < wp) ? (gtot - pre[w]) : -30.0f);
        }
        PipeBarrier<PIPE_V>();
        AscendC::Exp(expArgs, expArgs, 32);
        SyncV2S();

        // ---- s_q / s_k: sum_w d[w] * (kq/kk)[w] * rep[w] ----
        Duplicate(sQ, 0.0f, STEP_V);
        Duplicate(sK, 0.0f, STEP_V);
        PipeBarrier<PIPE_V>();
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
            // ---- phi / fs / u block loads ----
            LocalTensor<bfloat16_t> phiUb = inQPhi.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> fsUb = inQFs.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> uUb = inQU.AllocTensor<bfloat16_t>();
            // m <= P heads keep only the merged map (m*K); phi gains section
            // is skipped entirely (xq/xk are overridden by tq/tk)
            uint32_t phiLen = (m <= STEP_P)
                ? m * STEP_K
                : (STEP_P * STEP_K + (STEP_P + 1) * fg);
            DataCopyParams phiParams{1, static_cast<uint16_t>(phiLen * 2),
                                     0, 0};
            DataCopyPadParams phiPad{false, 0, 0, 0};
            DataCopyPad(phiUb,
                        gmPhi[(uint64_t)cidx * sSm + layPhi],
                        phiParams, phiPad);
            DataCopy(fsUb, gmFs[(uint64_t)cidx * sSf + layFs], STEP_W * fg);
            DataCopy(uUb, gmU[(uint64_t)cidx * sSu + layU], m * STEP_V);
            inQPhi.EnQue(phiUb);
            inQFs.EnQue(fsUb);
            inQU.EnQue(uUb);
            phiUb = inQPhi.DeQue<bfloat16_t>();
            fsUb = inQFs.DeQue<bfloat16_t>();
            uUb = inQU.DeQue<bfloat16_t>();
            PipeBarrier<PIPE_ALL>();
            Cast(fsF, fsUb, RoundMode::CAST_NONE, STEP_W * fg);
            PipeBarrier<PIPE_V>();
            if (mode == 26) {  // checkpoint: after phi/fs/u loads
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
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
                SyncV2S();
                tq[p] = redDst.GetValue(0);
                Mul(tmp, pivF, kF, STEP_K);
                PipeBarrier<PIPE_V>();
                ReduceSumHalfInterval(redDst, tmp, STEP_K);
                SyncV2S();
                tk[p] = redDst.GetValue(0);
            }
            if (mode == 27) {  // checkpoint: after pivots
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // gains projection into xq / xk (m-wide)
            Cast(adF, phiUb[STEP_P * STEP_K], RoundMode::CAST_NONE, m);
            PipeBarrier<PIPE_V>();
            if (mode == 32) {
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }
            Duplicate(xq, 0.0f, m);
            Duplicate(xk, 0.0f, m);
            PipeBarrier<PIPE_V>();
            if (mode == 33) {
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }
            for (uint32_t p = 0; p < STEP_P; ++p) {
                Cast(tmp2, phiUb[STEP_P * STEP_K + (p + 1) * fg],
                     RoundMode::CAST_NONE, m);
                PipeBarrier<PIPE_V>();
                if (mode == 35 && p == 0) {
                    inQPhi.FreeTensor(phiUb);
                    inQFs.FreeTensor(fsUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
                Muls(tmp, tmp2, tq[p], m);
                PipeBarrier<PIPE_V>();
                if (mode == 36 && p == 0) {
                    inQPhi.FreeTensor(phiUb);
                    inQFs.FreeTensor(fsUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
                Add(xq, xq, tmp, m);
                PipeBarrier<PIPE_V>();
                if (mode == 37 && p == 0) {
                    inQPhi.FreeTensor(phiUb);
                    inQFs.FreeTensor(fsUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
                Muls(tmp, tmp2, tk[p], m);
                PipeBarrier<PIPE_V>();
                Add(xk, xk, tmp, m);
                PipeBarrier<PIPE_V>();
                if (mode == 34 && p == 0) {
                    inQPhi.FreeTensor(phiUb);
                    inQFs.FreeTensor(fsUb);
                    inQU.FreeTensor(uUb);
                    CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                    return;
                }
            }
            if (mode == 29) {  // checkpoint: after gains p-loop
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
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
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
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
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // fcur = beta*(xk - ek); x = xq - eq - fcur*curKq
            Muls(tmp, ek, beta, m);
            PipeBarrier<PIPE_V>();
            Sub(xk, xk, tmp, m);
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
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
                inQU.FreeTensor(uUb);
                CheckpointEmit(n, hv, qIn, kIn, vIn, gateIn, ringKIn, ringDIn);
                return;
            }

            // fs[wp] = fcur (bf16, m elems)
            LocalTensor<bfloat16_t> fcurBf =
                calc[O_KN].ReinterpretCast<bfloat16_t>();
            Cast(fcurBf, fcur, RoundMode::CAST_RINT, m);
            SyncV2Mte3();
            DataCopyParams fsParams{1, static_cast<uint16_t>(m * 2), 0, 0};
            DataCopyPad(gmFs[(uint64_t)cidx * sSf + layFs + wp * fg],
                        fcurBf, fsParams);
            SyncMte32V();
            if (mode == 24) {  // checkpoint: after phi/fs/u math
                inQPhi.FreeTensor(phiUb);
                inQFs.FreeTensor(fsUb);
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
            inQPhi.FreeTensor(phiUb);
            inQFs.FreeTensor(fsUb);
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
            LocalTensor<bfloat16_t> ob = outBuf.Get<bfloat16_t>();
            Cast(ob, tmp, RoundMode::CAST_RINT, STEP_V);
            PipeBarrier<PIPE_V>();
            DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], ob,
                     STEP_V);
            inQQ.FreeTensor(qIn);
            inQK.FreeTensor(kIn);
            inQV.FreeTensor(vIn);
            inQGate.FreeTensor(gateIn);
            inQRingK.FreeTensor(ringKIn);
            inQRingD.FreeTensor(ringDIn);
            return;
        }

        // ---- d store + out store ----
        LocalTensor<bfloat16_t> ob = outBuf.Get<bfloat16_t>();
        if (wp == STEP_W - 1) {
            SyncV2Mte3();
            DataCopy(gmCurrentD[(uint64_t)cidx * hvNum * STEP_V + hv * STEP_V],
                     dc, STEP_V);
            SyncMte32V();
        } else {
            LocalTensor<bfloat16_t> dcBf =
                calc[O_HK2].ReinterpretCast<bfloat16_t>();
            Cast(dcBf, dc, RoundMode::CAST_RINT, STEP_V);
            SyncV2Mte3();
            DataCopy(gmDCache[(uint64_t)slot * sDcSlot + hv * sDcHead
                              + wp * STEP_V],
                     dcBf, STEP_V);
            SyncMte32V();
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
            Cast(ob, tmp2, RoundMode::CAST_RINT, STEP_V);
        } else {
            Cast(ob, tmp, RoundMode::CAST_RINT, STEP_V);
        }
        SyncV2Mte3();
        DataCopy(gmOut[(uint64_t)n * hvNum * STEP_V + hv * STEP_V], ob, STEP_V);
        SyncMte32V();

        inQQ.FreeTensor(qIn);
        inQK.FreeTensor(kIn);
        inQV.FreeTensor(vIn);
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
    TQue<QuePosition::VECIN, 1> inQRingK, inQRingD, inQPhi, inQFs,
        inQGate, inQQ, inQK, inQV;
    TQue<QuePosition::VECIN, 1> inQU;
    TBuf<QuePosition::VECCALC> calcBuf, outBuf;
    TBuf<QuePosition::VECCALC> metaSlots, metaWp, metaCidx, metaA, metaB;
    TBuf<QuePosition::VECCALC> metaALog, metaDt, metaRanks, metaLayout;
    LocalTensor<int32_t> slotsUb, cidxUb, ranksUb, layoutUb;
    LocalTensor<int32_t> wpUb;
    LocalTensor<float> aUb, bUb, alogUb, dtUb;
    static constexpr uint32_t STEP_V_MTE3_EVT = 0;
    static constexpr uint32_t STEP_MTE3_V_EVT = 1;
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
