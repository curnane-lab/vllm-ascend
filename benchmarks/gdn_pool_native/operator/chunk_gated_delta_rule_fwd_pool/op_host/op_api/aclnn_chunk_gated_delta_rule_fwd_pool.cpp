/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_gated_delta_rule_fwd_pool.h"

#include "chunk_gated_delta_rule_fwd_pool.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include <initializer_list>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr int64_t HEAD_DIM = 128;
constexpr int64_t CHUNK_SIZE = 64;

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

const aclTensor *TransposeContiguous(const aclTensor *tensor, std::initializer_list<int64_t> dims,
                                     aclOpExecutor *executor)
{
    const aclIntArray *perm = executor->AllocIntArray(dims.begin(), dims.size());
    if (perm == nullptr) {
        return nullptr;
    }
    const aclTensor *permuted = l0op::Transpose(tensor, perm, executor);
    if (permuted == nullptr) {
        return nullptr;
    }
    const aclTensor *materialized = l0op::Contiguous(permuted, executor);
    if (materialized == nullptr) {
        return nullptr;
    }
    const aclTensor *reshaped = l0op::Reshape(materialized, permuted->GetViewShape(), executor);
    if (reshaped != nullptr) {
        reshaped->SetStorageShape(reshaped->GetViewShape());
        reshaped->SetOriginalShape(reshaped->GetViewShape());
    }
    return reshaped;
}

aclnnStatus CheckMetadata(const aclIntArray *cu, const aclIntArray *chunkIndices,
                          int64_t tokens, int64_t chunkSize)
{
    CHECK_COND((cu == nullptr) == (chunkIndices == nullptr), ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be provided together.");
    if (cu == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(cu->Size() >= 2 && (*cu)[0] == 0 && (*cu)[cu->Size() - 1] == tokens,
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens must start at zero and end at total tokens.");
    int64_t expectedChunks = 0;
    for (size_t i = 0; i + 1 < cu->Size(); ++i) {
        CHECK_COND((*cu)[i] <= (*cu)[i + 1], ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be nondecreasing.");
        expectedChunks += ((*cu)[i + 1] - (*cu)[i] + chunkSize - 1) / chunkSize;
    }
    CHECK_COND(chunkIndices->Size() == expectedChunks * 2, ACLNN_ERR_PARAM_INVALID,
               "chunk_indices must have two entries per chunk.");
    size_t offset = 0;
    for (size_t sequence = 0; sequence + 1 < cu->Size(); ++sequence) {
        const int64_t chunks = ((*cu)[sequence + 1] - (*cu)[sequence] + chunkSize - 1) / chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND((*chunkIndices)[offset] == static_cast<int64_t>(sequence) &&
                           (*chunkIndices)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunk_indices must use canonical sequence-major order.");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckPool(const aclTensor *pool, const aclTensor *finalState,
                      const aclTensor *source, const aclTensor *destination,
                      const aclTensor *hasInitial, int64_t valueHeads, int64_t sequences)
{
    CHECK_COND(pool != nullptr && finalState != nullptr && source != nullptr &&
                   destination != nullptr && hasInitial != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "pool state, destination, and all slot tensors are required.");
    CHECK_COND(pool->GetDataType() == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "pool state must be bfloat16.");
    const auto shape = pool->GetViewShape();
    CHECK_COND(shape.GetDimNum() == 4 && shape.GetDim(1) == valueHeads &&
                   shape.GetDim(2) == HEAD_DIM && shape.GetDim(3) == HEAD_DIM,
               ACLNN_ERR_PARAM_INVALID, "pool state must have shape [P,HV,V,K] with V=K=128.");
    const auto &strides = pool->GetViewStrides();
    if (!(strides.size() == 4 && strides[3] == 1 && strides[2] == HEAD_DIM &&
          strides[1] >= HEAD_DIM * HEAD_DIM && strides[0] >= valueHeads * strides[1])) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "pool state requires contiguous V/K and nonoverlapping batch/head strides; "
                "stride_count=%zu, strides=[%lld,%lld,%lld,%lld].",
                strides.size(),
                strides.size() > 0 ? static_cast<long long>(strides[0]) : 0LL,
                strides.size() > 1 ? static_cast<long long>(strides[1]) : 0LL,
                strides.size() > 2 ? static_cast<long long>(strides[2]) : 0LL,
                strides.size() > 3 ? static_cast<long long>(strides[3]) : 0LL);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_COND(finalState->GetDataType() == pool->GetDataType() &&
                   finalState->GetViewShape() == shape && finalState->GetViewStrides() == strides,
               ACLNN_ERR_PARAM_INVALID, "finalStateOut must alias poolState with the same layout.");
    for (const aclTensor *indices : {source, destination}) {
        CHECK_COND(indices->GetDataType() == DataType::DT_INT32 &&
                       indices->GetViewShape().GetDimNum() == 1 &&
                       indices->GetViewShape().GetDim(0) == sequences,
                   ACLNN_ERR_PARAM_INVALID, "source/destination slots must be INT32 [sequenceCount].");
    }
    CHECK_COND(hasInitial->GetDataType() == DataType::DT_BOOL &&
                   hasInitial->GetViewShape().GetDimNum() == 1 &&
                   hasInitial->GetViewShape().GetDim(0) == sequences,
               ACLNN_ERR_PARAM_INVALID, "hasInitial must be BOOL [sequenceCount].");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                        const aclTensor *g, const aclTensor *beta, const aclTensor *oOut,
                        const aclIntArray *cu, const aclIntArray *chunkIndices, int64_t chunkSize)
{
    CHECK_COND(q != nullptr && k != nullptr && v != nullptr && g != nullptr && beta != nullptr && oOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q/k/v/g/beta/oOut must not be null.");
    CHECK_COND(q->GetDataType() == DataType::DT_BF16 && k->GetDataType() == DataType::DT_BF16 &&
                   v->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "pool I/O currently requires BF16 q/k/v.");
    CHECK_COND(g->GetDataType() == DataType::DT_FLOAT && beta->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "g must be FP32 and beta must be BF16.");
    CHECK_COND(chunkSize == CHUNK_SIZE, ACLNN_ERR_PARAM_INVALID, "pool I/O requires chunk_size=64.");
    const auto qShape = q->GetViewShape();
    const auto kShape = k->GetViewShape();
    const auto vShape = v->GetViewShape();
    CHECK_COND(qShape.GetDimNum() == 4 && kShape == qShape && vShape.GetDimNum() == 4 &&
                   qShape.GetDim(3) == HEAD_DIM && vShape.GetDim(3) == HEAD_DIM &&
                   vShape.GetDim(0) == qShape.GetDim(0) && vShape.GetDim(2) == qShape.GetDim(2) &&
                   vShape.GetDim(1) > 0 && vShape.GetDim(1) % qShape.GetDim(1) == 0,
               ACLNN_ERR_PARAM_INVALID,
               "q/k must be [B,HK,T,128], v must be [B,HV,T,128], and HK must divide HV.");
    for (const aclTensor *gate : {g, beta}) {
        const auto shape = gate->GetViewShape();
        CHECK_COND(shape.GetDimNum() == 3 && shape.GetDim(0) == qShape.GetDim(0) &&
                       shape.GetDim(1) == qShape.GetDim(2) && shape.GetDim(2) == vShape.GetDim(1),
                   ACLNN_ERR_PARAM_INVALID, "g/beta must be [B,T,HV].");
    }
    const auto outputShape = oOut->GetViewShape();
    CHECK_COND(oOut->GetDataType() == DataType::DT_BF16 && outputShape.GetDimNum() == 4 &&
                   outputShape.GetDim(0) == qShape.GetDim(0) && outputShape.GetDim(1) == qShape.GetDim(2) &&
                   outputShape.GetDim(2) == vShape.GetDim(1) && outputShape.GetDim(3) == HEAD_DIM,
               ACLNN_ERR_PARAM_INVALID, "oOut must be BF16 [B,T,HV,128].");
    return CheckMetadata(cu, chunkIndices, qShape.GetDim(2), chunkSize);
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnChunkGatedDeltaRuleFwdPoolGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *poolState,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const aclTensor *sourceSlots,
    const aclTensor *destinationSlots,
    const aclTensor *hasInitial,
    double scale,
    int64_t chunkSize,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be null.");
    CHECK_RET(CheckParams(q, k, v, g, beta, oOut, cuSeqlensOptional, chunkIndicesOptional, chunkSize) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    const int64_t batch = q->GetViewShape().GetDim(0);
    const int64_t tokens = q->GetViewShape().GetDim(2);
    const int64_t keyHeads = q->GetViewShape().GetDim(1);
    const int64_t valueHeads = v->GetViewShape().GetDim(1);
    const int64_t sequences = cuSeqlensOptional == nullptr ? batch : cuSeqlensOptional->Size() - 1;
    CHECK_RET(CheckPool(poolState, finalStateOut, sourceSlots, destinationSlots, hasInitial,
                        valueHeads, sequences) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(MakeContiguous(q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const aclTensor *betaFloat = l0op::Cast(beta, DataType::DT_FLOAT, executorPtr);
    CHECK_RET(betaFloat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *gBht = TransposeContiguous(g, {0, 2, 1}, executorPtr);
    const aclTensor *betaBht = TransposeContiguous(betaFloat, {0, 2, 1}, executorPtr);
    const auto matrixShape = MakeShape({batch, valueHeads, tokens, chunkSize});
    const aclTensor *aStorage = executorPtr->AllocTensor(matrixShape, DataType::DT_BF16, Format::FORMAT_ND);
    const aclTensor *aCompute = executorPtr->AllocTensor(matrixShape, DataType::DT_BF16, Format::FORMAT_ND);
    const aclTensor *gCumsum = executorPtr->AllocTensor(
        MakeShape({batch, tokens, valueHeads}), DataType::DT_FLOAT, Format::FORMAT_ND);
    const aclTensor *oHead = executorPtr->AllocTensor(
        MakeShape({batch, valueHeads, tokens, HEAD_DIM}), DataType::DT_BF16, Format::FORMAT_ND);
    CHECK_RET(gBht != nullptr && betaBht != nullptr && aStorage != nullptr && aCompute != nullptr &&
                  gCumsum != nullptr && oHead != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ChunkGatedDeltaRuleFwdPool(
        q, k, v, betaBht, aStorage, gBht, nullptr, poolState,
        cuSeqlensOptional, chunkIndicesOptional, sourceSlots, destinationSlots, hasInitial,
        true, chunkSize, scale, poolState->GetViewStrides()[1], poolState->GetViewStrides()[0],
        oHead, finalStateOut, gCumsum, aCompute, executorPtr);
    CHECK_RET(result[0] != nullptr && result[2] != nullptr && result[3] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *oSequence = TransposeContiguous(oHead, {0, 2, 1, 3}, executorPtr);
    CHECK_RET(oSequence != nullptr && l0op::ViewCopy(oSequence, oOut, executorPtr) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdPool(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleFwdPool);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkGatedDeltaRuleFwdPool launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
