#include "kernel_operator.h"

using namespace AscendC;

// pure vector-core kernel: keep the framework out of MIX codegen
ENABLE_FEATURE_FOR_COMPILE(default, KERNEL_TYPE_AIV_ONLY);

// LeapQuant GDN window flush for one (sequence, head) instance — the AscendC
// counterpart of the triton gdn_lq_flush kernel / torch GDNLeapQuantPools.flush.
// One AI vector core processes many instances sequentially; due instances
// (fill == L) rebuild the fp32 state S[V,K] in UB, fit R Compensator Tokens
// with one round of subspace iteration (modified Gram-Schmidt, projections
// applied twice), smooth and quantize the residual to int8, and reset the
// window bookkeeping. Fixed geometry: V=K=128, R=4, L=16, H=32 layout.
//
// Slot byte offsets (all 128-byte aligned, per-head strides fixed):
//   codes i8  [H,128,128] @ 0        kbuf bf16 [H,16,128] @ 524288
//   ubuf bf16 [H,16,128] @ 655360    u fp16 [H,4,128] @ 786432
//   q fp16    [H,4,128] @ 819200     s_k fp32 [H,128] @ 852992
//   s_v fp32  [H,128] @ 869376       w fp32 [H,16] @ 885760
//   p fp32    [H] @ 887808           fill i32 [H] @ 887936

constexpr uint32_t F_V = 128;
constexpr uint32_t F_K = 128;
constexpr uint32_t F_R = 4;
constexpr uint32_t F_L = 16;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr float QMAX = 127.0f;
constexpr float QEPS = 1e-12f;
constexpr float NORM_EPS = 1e-30f;

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

// per-head byte strides inside one slot
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

// dst[0] = sum of src[0..count); src is clobbered. Proven helper from
// gdn_sketch_step (canonical A3 halving-sum for count <= 128).
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
}

// halving max-reduce: leaves the max of src[0..count) in src[0..8); src is
// clobbered. Stops at 8 lanes so every operand stays 32-byte aligned (the
// final 8 -> 1 step is done with scalar reads after a V_S sync).
__aicore__ inline void ReduceMaxToEight(const LocalTensor<float>& src_local,
                                        int32_t count)
{
    int32_t bodyCount = count;
    while (bodyCount > 8) {
        bodyCount = bodyCount / 2;
        Max(src_local, src_local, src_local[bodyCount], bodyCount);
    }
}

class GdnLqFlushKernel {
public:
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR idx, GM_ADDR q0,
                                uint32_t myStart, uint32_t myCount, uint32_t t,
                                uint32_t h, int64_t sSlot, TPipe* pipe)
    {
        this->myStart = myStart;
        this->myCount = myCount;
        this->batch = t;
        this->hNum = h;
        this->sSlot = sSlot;
        this->stateRaw = reinterpret_cast<__gm__ uint8_t*>(state);

        gmIdx.SetGlobalBuffer((__gm__ int64_t*)idx);
        gmQ0.SetGlobalBuffer((__gm__ float*)q0);

        auto align32 = [](uint32_t b) { return (b + 31) / 32 * 32; };
        pipe->InitBuffer(sBuf, F_V * F_K * sizeof(float));
        pipe->InitBuffer(codesBuf, F_V * F_K * sizeof(int8_t));
        pipe->InitBuffer(kbufRaw, F_L * F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(ubufRaw, F_L * F_K * sizeof(bfloat16_t));
        pipe->InitBuffer(uqRaw, 2 * F_R * F_V * sizeof(half));
        pipe->InitBuffer(kbufF, F_L * F_K * sizeof(float));
        pipe->InitBuffer(ubufF, F_L * F_K * sizeof(float));
        pipe->InitBuffer(uF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(qF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(skF, F_K * sizeof(float));
        pipe->InitBuffer(svF, F_V * sizeof(float));
        pipe->InitBuffer(wF, align32(F_L * sizeof(float)));
        pipe->InitBuffer(q0F, F_R * F_V * sizeof(float));
        pipe->InitBuffer(qkF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(qvF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(ukF, F_R * F_V * sizeof(float));
        pipe->InitBuffer(sumAbs, F_K * sizeof(float));
        pipe->InitBuffer(scrBuf, 4 * F_K * sizeof(float));
        pipe->InitBuffer(tmpBuf, 2 * F_K * sizeof(float));
        pipe->InitBuffer(halfBuf, F_K * sizeof(half));
        pipe->InitBuffer(redBuf, 16 * 8 * sizeof(float));
        pipe->InitBuffer(metaBuf, 2 * align32(h * sizeof(int32_t)));
        pipe->InitBuffer(idxBuf, align32(t * sizeof(int64_t)));
        pipe->InitBuffer(outQ, 1, F_V * F_K * sizeof(int8_t) + 3584);

        sUb = sBuf.Get<float>();
        codesUb = codesBuf.Get<int8_t>();
        kbufRawUb = kbufRaw.Get<bfloat16_t>();
        ubufRawUb = ubufRaw.Get<bfloat16_t>();
        uqRawUb = uqRaw.Get<half>();
        kbufFUb = kbufF.Get<float>();
        ubufFUb = ubufF.Get<float>();
        uFUb = uF.Get<float>();
        qFUb = qF.Get<float>();
        skFUb = skF.Get<float>();
        svFUb = svF.Get<float>();
        wFUb = wF.Get<float>();
        q0FUb = q0F.Get<float>();
        qkFUb = qkF.Get<float>();
        qvFUb = qvF.Get<float>();
        ukFUb = ukF.Get<float>();
        sumAbsUb = sumAbs.Get<float>();
        tmpUb = tmpBuf.Get<float>();
        tmp2Ub = tmpBuf.Get<float>()[F_K];
        scrUb = scrBuf.Get<float>();
        redUb = redBuf.Get<float>();
        halfUb = halfBuf.Get<half>();
        fillUb = metaBuf.Get<int32_t>();
        pUb = metaBuf.Get<float>()[align32(h * sizeof(int32_t)) / sizeof(float)];
        idxUb = idxBuf.Get<int64_t>();

        // exact-length idx load (any batch) from the tensor base
        DataCopyPadParams idxPad{false, 0, 0, 0};
        DataCopyPad(idxUb, gmIdx[0],
                    DataCopyParams{1, static_cast<uint16_t>(batch * 8), 0, 0},
                    idxPad);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < myCount; ++i) {
            ProcessOne(myStart + i);
        }
    }

private:
    // V->S: scalar (S-pipe) GetValue reads of V-computed values are NOT
    // ordered by --cce-auto-sync; a manual V_S event is required once per
    // batch of scalar reads (same constant-id pattern as gdn_sketch_step).
    __aicore__ inline void SyncV2S()
    {
        SetFlag<HardEvent::V_S>(FLUSH_V_S_EVT);
        WaitFlag<HardEvent::V_S>(FLUSH_V_S_EVT);
    }

    __aicore__ inline float DotV(const LocalTensor<float>& a,
                                 const LocalTensor<float>& b)
    {
        Mul(tmpUb, a, b, F_K);
        ReduceSumHalfInterval(redUb, tmpUb, F_K);
        SyncV2S();
        return redUb.GetValue(0);
    }

    // modified Gram-Schmidt on 4 [128] vectors held contiguously in vec
    // (vec[r*128]); projections against each previous column applied twice,
    // each column normalized before later columns use it. Matches the torch
    // _mgs reference exactly.
    __aicore__ inline void Mgs4(const LocalTensor<float>& vec)
    {
        float d = DotV(vec, vec);
        float inv = 1.0f / sqrt(d >= NORM_EPS ? d : NORM_EPS);
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
            float invr = 1.0f / sqrt(dr >= NORM_EPS ? dr : NORM_EPS);
            Muls(vr, vr, invr, F_K);
        }
    }

    __aicore__ inline void ProcessOne(uint32_t inst)
    {
        uint32_t b = inst / hNum;
        uint32_t h = inst % hNum;
        // low 32-bit word of the int64 slot id (little-endian word order,
        // same assumption as gdn_sketch_step's write_pos handling)
        int32_t slot = idxUb.ReinterpretCast<int32_t>().GetValue(b * 2);
        __gm__ uint8_t* base = stateRaw
            + static_cast<uint64_t>(slot) * static_cast<uint64_t>(sSlot);

        GlobalTensor<int32_t> gmFill;
        gmFill.SetGlobalBuffer((__gm__ int32_t*)(base + OFF_FILL));
        GlobalTensor<float> gmP;
        gmP.SetGlobalBuffer((__gm__ float*)(base + OFF_P));
        // per-head window bookkeeping first: non-due heads exit before any
        // heavy load (fill/p region is 128-byte aligned, H*4 = 128B)
        DataCopy(fillUb, gmFill[0], hNum);
        DataCopy(pUb, gmP[0], hNum);
        PipeBarrier<PIPE_ALL>();
        int32_t fill = fillUb.GetValue(h);
        if (fill < static_cast<int32_t>(F_L)) {
            return;
        }
        float pval = pUb.GetValue(h);

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

        DataCopy(codesUb, gmCodes[0], F_V * F_K);
        DataCopy(kbufRawUb, gmKbuf[0], F_L * F_K);
        DataCopy(ubufRawUb, gmUbuf[0], F_L * F_K);
        DataCopy(uqRawUb, gmU[0], F_R * F_V);
        DataCopy(uqRawUb[F_R * F_V], gmQ[0], F_R * F_V);
        DataCopy(skFUb, gmSk[0], F_K);
        DataCopy(svFUb, gmSv[0], F_V);
        DataCopy(wFUb, gmW[0], F_L);
        DataCopy(q0FUb, gmQ0[0], F_R * F_V);
        PipeBarrier<PIPE_ALL>();

        // bf16/fp16 -> fp32
        Cast(kbufFUb, kbufRawUb, RoundMode::CAST_NONE, F_L * F_K);
        Cast(ubufFUb, ubufRawUb, RoundMode::CAST_NONE, F_L * F_K);
        Cast(uFUb, uqRawUb, RoundMode::CAST_NONE, F_R * F_V);
        Cast(qFUb, uqRawUb[F_R * F_V], RoundMode::CAST_NONE, F_R * F_V);
        // kw[j] = w[j] * kbuf[j]
        for (uint32_t j = 0; j < F_L; ++j) {
            float wj = wFUb.GetValue(j);
            Muls(kbufFUb[j * F_K], kbufFUb[j * F_K], wj, F_K);
        }
        Duplicate(qkFUb, 0.0f, F_R * F_V);
        // one V_S pair covers every scalar read below (all source tensors
        // were last written by the casts/reduces above)
        SyncV2S();

        // ---- pass 0+1: rebuild S row by row and accumulate m1 = S^T q0 ----
        float pOverQmax = pval / QMAX;
        for (uint32_t v = 0; v < F_V; ++v) {
            LocalTensor<float> srow = sUb[v * F_K];
            float av = pOverQmax * svFUb.GetValue(v);
            // no direct int8<->float cast intrinsic on this arch: route the
            // exact integer values through fp16 (integers <= 127 are exact)
            Cast(halfUb, codesUb[v * F_K], RoundMode::CAST_NONE, F_K);
            Cast(srow, halfUb, RoundMode::CAST_NONE, F_K);
            Mul(srow, srow, skFUb, F_K);
            Muls(srow, srow, av, F_K);
            for (uint32_t r = 0; r < F_R; ++r) {
                // comp-token term carries the same p factor as the codes
                // term (torch: S = p * s0 + window)
                float qrv = pval * qFUb.GetValue(r * F_V + v);
                Muls(tmpUb, uFUb[r * F_K], qrv, F_K);
                Add(srow, srow, tmpUb, F_K);
            }
            for (uint32_t j = 0; j < F_L; ++j) {
                float ub = ubufFUb.GetValue(j * F_V + v);
                Muls(tmpUb, kbufFUb[j * F_K], ub, F_K);
                Add(srow, srow, tmpUb, F_K);
            }
            for (uint32_t r = 0; r < F_R; ++r) {
                float q0v = q0FUb.GetValue(r * F_V + v);
                Muls(tmpUb, srow, q0v, F_K);
                Add(qkFUb[r * F_K], qkFUb[r * F_K], tmpUb, F_K);
            }
        }

        // ---- subspace iteration: qk = mgs(S^T q0) ----
        Mgs4(qkFUb);

        // ---- pass 2: qv_r[v] = sum_k S[v,k] qk_r[k], then mgs ----
        // WholeReduceSum dst must be 32B-aligned: reduce into red slots
        // (stride 8 floats), four rows (16 dots) per V_S round trip, then
        // scatter with SetValue (V-pipe scalar store at any offset).
        for (uint32_t v0 = 0; v0 < F_V; v0 += 4) {
            uint32_t idx = 0;
            for (uint32_t i = 0; i < 4; ++i) {
                LocalTensor<float> srow = sUb[(v0 + i) * F_K];
                for (uint32_t r = 0; r < F_R; ++r) {
                    Mul(tmpUb, srow, qkFUb[r * F_K], F_K);
                    ReduceSumHalfInterval(redUb[idx * 8], tmpUb, F_K);
                    ++idx;
                }
            }
            SyncV2S();
            idx = 0;
            for (uint32_t i = 0; i < 4; ++i) {
                for (uint32_t r = 0; r < F_R; ++r) {
                    qvFUb.SetValue(r * F_V + v0 + i, redUb[idx * 8].GetValue(0));
                    ++idx;
                }
            }
        }
        Mgs4(qvFUb);

        // ---- pass 3: uk_r[k] = sum_v S[v,k] qv_r[v] ----
        Duplicate(ukFUb, 0.0f, F_R * F_V);
        SyncV2S();  // qv reduce results -> scalar reads
        for (uint32_t v = 0; v < F_V; ++v) {
            LocalTensor<float> srow = sUb[v * F_K];
            float c0 = qvFUb.GetValue(v);
            float c1 = qvFUb.GetValue(F_V + v);
            float c2 = qvFUb.GetValue(2 * F_V + v);
            float c3 = qvFUb.GetValue(3 * F_V + v);
            Muls(tmpUb, srow, c0, F_K);
            Add(ukFUb[0], ukFUb[0], tmpUb, F_K);
            Muls(tmpUb, srow, c1, F_K);
            Add(ukFUb[F_K], ukFUb[F_K], tmpUb, F_K);
            Muls(tmpUb, srow, c2, F_K);
            Add(ukFUb[2 * F_K], ukFUb[2 * F_K], tmpUb, F_K);
            Muls(tmpUb, srow, c3, F_K);
            Add(ukFUb[3 * F_K], ukFUb[3 * F_K], tmpUb, F_K);
        }

        // ---- pass 4a: E = S - qv uk^T (in place), sk = sqrt(mean_v |E|) ----
        Duplicate(sumAbsUb, 0.0f, F_K);
        for (uint32_t v = 0; v < F_V; ++v) {
            LocalTensor<float> srow = sUb[v * F_K];
            float c0 = qvFUb.GetValue(v);
            float c1 = qvFUb.GetValue(F_V + v);
            float c2 = qvFUb.GetValue(2 * F_V + v);
            float c3 = qvFUb.GetValue(3 * F_V + v);
            Muls(tmpUb, ukFUb[0], c0, F_K);
            Muls(tmp2Ub, ukFUb[F_K], c1, F_K);
            Add(tmpUb, tmpUb, tmp2Ub, F_K);
            Muls(tmp2Ub, ukFUb[2 * F_K], c2, F_K);
            Add(tmpUb, tmpUb, tmp2Ub, F_K);
            Muls(tmp2Ub, ukFUb[3 * F_K], c3, F_K);
            Add(tmpUb, tmpUb, tmp2Ub, F_K);
            Sub(srow, srow, tmpUb, F_K);
            Abs(tmpUb, srow, F_K);
            Add(sumAbsUb, sumAbsUb, tmpUb, F_K);
        }
        // sk into skFUb (old s_k dead after the rebuild)
        Muls(skFUb, sumAbsUb, 0.0078125f, F_K);  // 1/V
        Sqrt(skFUb, skFUb, F_K);
        Maxs(skFUb, skFUb, QEPS, F_K);

        // ---- pass 4b+5 fused: sv[v] = max_k |E|/sk, codes = rint(127 E/sk/sv) ----
        // four rows per V_S round trip: the |E|/sk rows are reduced to one
        // max lane each, a single sync makes all four readable, then the
        // quantized rows are produced.
        for (uint32_t v0 = 0; v0 < F_V; v0 += 4) {
            for (uint32_t i = 0; i < 4; ++i) {
                LocalTensor<float> acc = scrUb[i * F_K];
                Div(acc, sUb[(v0 + i) * F_K], skFUb, F_K);
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
                svFUb.SetValue(v, svv);
                float inv = QMAX / svv;
                Div(tmpUb, sUb[v * F_K], skFUb, F_K);
                Muls(tmpUb, tmpUb, inv, F_K);
                Maxs(tmpUb, tmpUb, -QMAX, F_K);
                Mins(tmpUb, tmpUb, QMAX, F_K);
                // round-half-even in fp32, then two exact narrowing casts
                Cast(halfUb, tmpUb, RoundMode::CAST_RINT, F_K);
                Cast(codesUb[v * F_K], halfUb, RoundMode::CAST_RINT, F_K);
            }
        }

        // ---- write back: stage all payloads into ONE queue buffer (the
        // framework V->MTE3 ordering is paid once), then fan out.
        // Payload layout (outQ buffer, int8 view):
        //   [0:16384]       codes int8
        //   [16384:16896]   s_k fp32 (f32 offset 4096)
        //   [16896:17408]   s_v fp32 (f32 offset 4224)
        //   [17408:18432]   u fp16, 512 halves (half offset 8704)
        //   [18432:19456]   q fp16, 512 halves (half offset 9216)
        //   [19456:19520]   w zeros (f32 offset 4864) -- MUST sit after q:
        //                     the old offset 4352 overlapped u and clobbered
        //                     the first 48 halves of compensator token r=0
        //   [19520:19524]   p = 1.0 (f32 offset 4880)
        //   [19524:19528]   fill = 0 (i32 view of the w-zero region)
        LocalTensor<int8_t> oq = outQ.AllocTensor<int8_t>();
        DataCopy(oq, codesUb, F_V * F_K);
        LocalTensor<float> sf = oq.ReinterpretCast<float>();
        DataCopy(sf[F_V * F_K / 4], skFUb, F_K);
        DataCopy(sf[F_V * F_K / 4 + F_K], svFUb, F_V);
        LocalTensor<half> uh = oq.ReinterpretCast<half>();
        Cast(uh[F_R * F_V * 17], ukFUb, RoundMode::CAST_RINT, F_R * F_V);
        Cast(uh[F_R * F_V * 18], qvFUb, RoundMode::CAST_RINT, F_R * F_V);
        Duplicate(sf[F_V * F_K / 4 + 6 * F_K], 0.0f, F_L);
        Duplicate(sf[F_V * F_K / 4 + 6 * F_K + 16], 1.0f, 8);
        outQ.EnQue(oq);
        LocalTensor<int8_t> dq = outQ.DeQue<int8_t>();
        DataCopy(gmCodes[0], dq, F_V * F_K);
        LocalTensor<float> df = dq.ReinterpretCast<float>();
        DataCopy(gmSk[0], df[F_V * F_K / 4], F_K);
        DataCopy(gmSv[0], df[F_V * F_K / 4 + F_K], F_V);
        LocalTensor<half> dh = dq.ReinterpretCast<half>();
        DataCopy(gmU[0], dh[F_R * F_V * 17], F_R * F_V);
        DataCopy(gmQ[0], dh[F_R * F_V * 18], F_R * F_V);
        DataCopy(gmW[0], df[F_V * F_K / 4 + 6 * F_K], F_L);
        DataCopyPad(gmP[h], df[F_V * F_K / 4 + 6 * F_K + 16],
                    DataCopyParams{1, 4, 0, 0});
        // float 0.0f bit pattern is int32 0: read the w-zero region low word
        DataCopyPad(gmFill[h],
                    df.ReinterpretCast<int32_t>()[F_V * F_K / 4 + 6 * F_K],
                    DataCopyParams{1, 4, 0, 0});
        outQ.FreeTensor(dq);
    }

    GlobalTensor<int64_t> gmIdx;
    GlobalTensor<float> gmQ0;
    __gm__ uint8_t* stateRaw = nullptr;
    TBuf<QuePosition::VECCALC> sBuf, codesBuf, kbufRaw, ubufRaw, uqRaw;
    TBuf<QuePosition::VECCALC> kbufF, ubufF, uF, qF, skF, svF, wF;
    TBuf<QuePosition::VECCALC> q0F, qkF, qvF, ukF, sumAbs, scrBuf, tmpBuf;
    TBuf<QuePosition::VECCALC> redBuf, halfBuf;
    TBuf<QuePosition::VECCALC> metaBuf, idxBuf;
    TQue<QuePosition::VECOUT, 1> outQ;
    LocalTensor<float> sUb, kbufFUb, ubufFUb, uFUb, qFUb, skFUb, svFUb;
    LocalTensor<float> wFUb, q0FUb, qkFUb, qvFUb, ukFUb, sumAbsUb, tmpUb;
    LocalTensor<float> tmp2Ub, scrUb, redUb, pUb;
    LocalTensor<half> halfUb;
    LocalTensor<int8_t> codesUb;
    LocalTensor<bfloat16_t> kbufRawUb, ubufRawUb;
    LocalTensor<half> uqRawUb;
    LocalTensor<int32_t> fillUb;
    LocalTensor<int64_t> idxUb;
    static constexpr uint32_t FLUSH_V_S_EVT = 2;
    uint32_t myStart = 0, myCount = 0, batch = 0, hNum = 32;
    int64_t sSlot = 0;
};

extern "C" __global__ __aicore__ void gdn_lq_flush(
    GM_ADDR state, GM_ADDR idx, GM_ADDR q0, GM_ADDR out,
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
    GdnLqFlushKernel op;
    op.Init(state, idx, q0, myStart, myCount, tilingData.batch, tilingData.h,
            tilingData.sSlot, &pipe);
    op.Process();
}
