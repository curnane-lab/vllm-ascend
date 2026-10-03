#include "kernel_operator.h"

using namespace AscendC;

// M0 probe: per-instance ring staging + fused q-dots + gate replay weights +
// V-wide d-replay accumulation — the compute core of the SketchSSM step
// kernel. W=16, K=128, V=128 fixed.
// mode (attr): 1 = staging only; 2 = +casts/norm; 3 = +ring dots/exp;
//              4 = +d-replay/out (full).


// Canonical A3 fp32 row-sum helpers (from csrc/moe/add_rms_norm_bias).
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;

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

constexpr uint32_t PROBE_W = 16;
constexpr uint32_t PROBE_K = 128;
constexpr uint32_t PROBE_V = 128;

class GdnSketchProbeKernel {
public:
    __aicore__ inline GdnSketchProbeKernel() {}

    __aicore__ inline void Init(GM_ADDR ring, GM_ADDR q, GM_ADDR dring,
                                GM_ADDR gate, GM_ADDR out,
                                uint32_t numInstances, uint32_t myStart,
                                uint32_t myCount, uint32_t mode, TPipe* pipe)
    {
        gmRing.SetGlobalBuffer((__gm__ bfloat16_t*)ring);
        gmQ.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        gmDring.SetGlobalBuffer((__gm__ bfloat16_t*)dring);
        gmGate.SetGlobalBuffer((__gm__ float*)gate);
        gmOut.SetGlobalBuffer((__gm__ float*)out);
        this->myStart = myStart;
        this->myCount = myCount;
        this->mode = mode;

        pipe->InitBuffer(inQueueRing, 2, PROBE_W * PROBE_K * sizeof(bfloat16_t));
        pipe->InitBuffer(inQueueQ, 2, PROBE_K * sizeof(bfloat16_t));
        pipe->InitBuffer(inQueueD, 2, PROBE_W * PROBE_V * sizeof(bfloat16_t));
        pipe->InitBuffer(inQueueGate, 2, PROBE_W * sizeof(float));
        pipe->InitBuffer(calcBuf, (PROBE_K * 2 + PROBE_W * PROBE_K
            + PROBE_W * PROBE_V + 2 * PROBE_V + 64) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }

private:
    __aicore__ inline void ProcessOne(uint64_t inst)
    {
        LocalTensor<bfloat16_t> ringUb = inQueueRing.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> qUb = inQueueQ.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> dUb = inQueueD.AllocTensor<bfloat16_t>();
        LocalTensor<float> gateUb = inQueueGate.AllocTensor<float>();

        DataCopy(ringUb, gmRing[inst * PROBE_W * PROBE_K], PROBE_W * PROBE_K);
        DataCopy(qUb, gmQ[inst * PROBE_K], PROBE_K);
        DataCopy(dUb, gmDring[inst * PROBE_W * PROBE_V], PROBE_W * PROBE_V);
        DataCopy(gateUb, gmGate[inst * PROBE_W], PROBE_W);
        inQueueRing.EnQue(ringUb);
        inQueueQ.EnQue(qUb);
        inQueueD.EnQue(dUb);
        inQueueGate.EnQue(gateUb);

        LocalTensor<bfloat16_t> ringIn = inQueueRing.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> qIn = inQueueQ.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> dIn = inQueueD.DeQue<bfloat16_t>();
        LocalTensor<float> gateIn = inQueueGate.DeQue<float>();

        constexpr uint32_t OFF_RING = PROBE_K;
        constexpr uint32_t OFF_D = OFF_RING + PROBE_W * PROBE_K;
        constexpr uint32_t OFF_TMP = OFF_D + PROBE_W * PROBE_V;
        constexpr uint32_t OFF_ACC = OFF_TMP + PROBE_V;
        constexpr uint32_t OFF_RED = OFF_ACC + PROBE_V;
        constexpr uint32_t OFF_EXP = OFF_RED + PROBE_K;
        LocalTensor<float> qF = calcBuf.Get<float>();
        LocalTensor<float> ringF = calcBuf.Get<float>()[OFF_RING];
        LocalTensor<float> dF = calcBuf.Get<float>()[OFF_D];
        LocalTensor<float> tmp = calcBuf.Get<float>()[OFF_TMP];
        LocalTensor<float> acc = calcBuf.Get<float>()[OFF_ACC];
        LocalTensor<float> redDst = calcBuf.Get<float>()[OFF_RED];
        LocalTensor<float> expArgs = calcBuf.Get<float>()[OFF_EXP];

        if (mode == 0) {
            Duplicate(acc, 2.0f, PROBE_V);
            PipeBarrier<PIPE_V>();
        }
        if (mode >= 2) {
            Cast(qF, qIn, RoundMode::CAST_NONE, PROBE_K);
            Cast(ringF, ringIn, RoundMode::CAST_NONE, PROBE_W * PROBE_K);
            Cast(dF, dIn, RoundMode::CAST_NONE, PROBE_W * PROBE_V);
            PipeBarrier<PIPE_V>();

            Mul(tmp, qF, qF, PROBE_K);
            PipeBarrier<PIPE_V>();
            ReduceSumHalfInterval(redDst, tmp, PROBE_K);
            float qn = 1.0f / sqrt(redDst.GetValue(0) + 1e-6f);
            Muls(qF, qF, qn, PROBE_K);
            PipeBarrier<PIPE_V>();
        }

        if (mode == 2) {  // diag: emit qn
            DataCopy(gmOut[inst * PROBE_V], qF, PROBE_V);
            PipeBarrier<PIPE_V>();
            inQueueRing.FreeTensor(ringIn);
            inQueueQ.FreeTensor(qIn);
            inQueueD.FreeTensor(dIn);
            inQueueGate.FreeTensor(gateIn);
            return;
        }

        if (mode >= 3) {
            float suffix[PROBE_W + 1];
            suffix[PROBE_W] = 0.0f;
            for (int32_t w = PROBE_W - 1; w >= 0; --w) {
                suffix[w] = suffix[w + 1] + gateIn.GetValue(w);
            }
            float kqs[PROBE_W];
            for (uint32_t w = 0; w < PROBE_W; ++w) {
                Mul(tmp, ringF[w * PROBE_K], qF, PROBE_K);
                PipeBarrier<PIPE_V>();
                ReduceSumHalfInterval(redDst, tmp, PROBE_K);
                kqs[w] = redDst.GetValue(0);
            }
            for (uint32_t w = 0; w < PROBE_W; ++w) {
                expArgs.SetValue(w, suffix[w + 1]);
            }
            expArgs.SetValue(PROBE_W, suffix[0]);
            PipeBarrier<PIPE_V>();
            AscendC::Exp(expArgs, expArgs, 64);
            PipeBarrier<PIPE_V>();

            if (mode == 3) {  // diag: emit kq_s (w) and rep (V - W)
                for (uint32_t w = 0; w < PROBE_W; ++w) {
                    expArgs.SetValue(w, kqs[w]);
                }
                DataCopy(gmOut[inst * PROBE_V], expArgs, PROBE_V);
                PipeBarrier<PIPE_V>();
                inQueueRing.FreeTensor(ringIn);
                inQueueQ.FreeTensor(qIn);
                inQueueD.FreeTensor(dIn);
                inQueueGate.FreeTensor(gateIn);
                return;
            }

            if (mode >= 4) {
                Duplicate(acc, 0.0f, PROBE_V);
                PipeBarrier<PIPE_V>();
                for (uint32_t w = 0; w < PROBE_W; ++w) {
                    float wgt = kqs[w] * expArgs.GetValue(w);
                    Muls(tmp, dF[w * PROBE_V], wgt, PROBE_V);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, tmp, PROBE_V);
                    PipeBarrier<PIPE_V>();
                }
                Muls(acc, acc, expArgs.GetValue(PROBE_W), PROBE_V);
                PipeBarrier<PIPE_V>();
            }
        }



        DataCopy(gmOut[inst * PROBE_V], acc, PROBE_V);

        inQueueRing.FreeTensor(ringIn);
        inQueueQ.FreeTensor(qIn);
        inQueueD.FreeTensor(dIn);
        inQueueGate.FreeTensor(gateIn);
    }

    GlobalTensor<bfloat16_t> gmRing, gmQ, gmDring;
    GlobalTensor<float> gmGate, gmOut;
    TQue<QuePosition::VECIN, 2> inQueueRing, inQueueQ, inQueueD, inQueueGate;
    TBuf<QuePosition::VECCALC> calcBuf;
    uint32_t myStart = 0;
    uint32_t myCount = 0;
    uint32_t mode = 99;
};

extern "C" __global__ __aicore__ void gdn_sketch_probe(
    GM_ADDR ring, GM_ADDR q, GM_ADDR dring, GM_ADDR gate, GM_ADDR out,
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
        myStart = tilingData.remainderCores * (tilingData.instancesPerCore + 1)
            + (GetBlockIdx() - tilingData.remainderCores)
            * tilingData.instancesPerCore;
        myCount = tilingData.instancesPerCore;
    }
    TPipe pipe;
    GdnSketchProbeKernel op;
    op.Init(ring, q, dring, gate, out, tilingData.numInstances,
            myStart, myCount, tilingData.mode, &pipe);
    op.Process();
}
